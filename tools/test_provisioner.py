#!/usr/bin/env python3
"""Self-check: a provisioned pod must be internally consistent, per profile.

Drives the shared runtime's provisioner (comfyui-runtime/src/provisioner.py,
pinned by pins.json) against this repo's REAL template.json,
models_registry.json and workflows/ across every supported `minimax_quant`
value, case/whitespace normalization, and invalid-value fallback. For each, the
workflows copied to the user's ComfyUI must declare exactly the model files the
download manifest pulled: in the loader widgets AND in each node's
`properties.models`, which is what the ComfyUI frontend's "Missing Models"
dialog reads. A mismatch there tells the customer a file is missing and
offers to download it to their own PC. It also checks the source workflow
links and the tuned loader/scheduler defaults before the provisioner rewrites
copies for another quant profile.

`download_minimax_h3` installs only
workflows/minimaxH3ReferenceToVideo8_v10_cached.json and queues that graph's
models. The other workflow files stay in the repo, but their models are not
downloaded. The NSFW LoRA named in that graph is listed in auto_download so
it is neither fetched nor treated as a missing registry model.

Run: python3 tools/test_provisioner.py
Stdlib only, no pytest. Needs template.json + pins.json in the repo root.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_models import runtime_dir  # noqa: E402

TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
BF16_MODELS = {
    "ref2va": "minimax_h3_ref2va_bf16.safetensors",
}
CACHED_WORKFLOW = "minimaxH3ReferenceToVideo8_v10_cached.json"
IGNORED_LORA = "HMNSFW-AIO-V2.5.safetensors"
# Models the cached graph loads that are not quant-swapped.
CACHED_STATIC_MODELS = {
    "minimax_h3_audio_vae_fp32.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    "taeh3.safetensors",
}

# The refreshed workflows use the two 8-step 768p builds. The 4-step v1.2
# build is bundled as a manual dropdown alternative. All three stay explicit
# in template.json's extra_models contract.
BUNDLED_LORAS = [
    "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
]
TURBO_LORAS = BUNDLED_LORAS
REMOVED_TURBO_LORAS = {
    "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors",
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
}
LATENT_UPSCALER = "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
LATENT_UPSCALER_URL = (
    "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/"
    "main/minimax_h3_latent_upscaler_3d_conv_v1/"
    "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
)
LATENT_UPSCALER_NODE = (
    "https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git|"
    "d7c01b9011f2e8439493f6c02c29995a27df276f"
)
HYPERFLOW_FULL = "minimax_h3_hyperflow_8step_v1.0_comfyui_bf16.safetensors"
HYPERFLOW_PRUNED = "minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16.safetensors"
HYPERFLOW_NODE_WEIGHT = "custom_node_hyperflow_8step_v1.0_comfyui.safetensors"
HYPERFLOW_NODE = (
    "https://github.com/jalberty2018/ComfyUI-Hyperflow.git|"
    "90fac72fe147007c6fdfdeb10d5b9068f8eaf919"
)
HYPERFLOW_NODE_URL = (
    "https://huggingface.co/drbaph/Hyperflow-Comfyui/resolve/"
    "2cea953918c1f800e4d6d15304d65947b220ddc7/"
    + HYPERFLOW_NODE_WEIGHT
)
HYPERFLOW_URL_BASE = (
    "https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/"
    "bb2bc497cbaca89dadd0bcf1856eed4f8275be20/"
)
FL2VA_INT8 = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA_INT8 = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
FL2VA_TURBO_8STEP = (
    "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
)
REF2VA_TURBO_8STEP = (
    "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
)
SOURCE_WORKFLOW_DEFAULTS = {
    "MiniMax - I2V - Auto Prompt.json":
        (FL2VA_INT8, FL2VA_TURBO_8STEP, 0.8, True),
    "MiniMax - I2V - Custom Prompt.json":
        (FL2VA_INT8, FL2VA_TURBO_8STEP, 0.8, True),
    "MiniMax - R2V - Auto Prompt.json":
        (REF2VA_INT8, REF2VA_TURBO_8STEP, 0.85, False),
    "MiniMax - T2V - Auto Prompt.json":
        (FL2VA_INT8, FL2VA_TURBO_8STEP, 0.8, False),
    "MiniMax - T2V - Custom Prompt.json":
        (FL2VA_INT8, FL2VA_TURBO_8STEP, 0.8, False),
}
UPSCALING_WORKFLOWS = {
    "MiniMax - I2V - Auto Prompt - Upscaling.json",
    "MiniMax - I2V - Custom Prompt - Upscaling.json",
    "MiniMax - R2V - Auto Prompt - Upscaling.json",
    "MiniMax - T2V - Auto Prompt - Upscaling.json",
    "MiniMax - T2V - Custom Prompt - Upscaling.json",
}
REFMOD_NODE = "https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod.git|22ca77e247375dd47fc7340286b5f391887d3410"
REFMOD_WORKFLOWS = {"MiniMax - RefMod.json"} | {f"RefMod Studio/RefMod - {kind}.json" for kind in ("Images", "Video", "Audio")}

CHARACTER_LORA_PLACEHOLDER = "Your_Character_LoRA_Here.safetensors"

# label, minimax_quant, expected profile, expected warning fragment. The false
# profile selects full bf16; invalid values keep booting on int8 with a warning.
CASES = [
    ("unset", None, "int8", None),
    ("int8", "int8", "int8", None),
    ("fp8", "fp8", "fp8", None),
    ("FP8", "FP8", "fp8", None),
    ("nvfp4", "nvfp4", "nvfp4", None),
    ("false", "false", "false", None),
    ("bf16", "bf16", "bf16", None),
    ("BF16", "BF16", "bf16", None),
    ("FALSE", "FALSE", "false", None),
    ("false-whitespace", " false ", "false", None),
    ("invalid-quant", "not-a-quant", "int8",
     "warning: unknown minimax_quant"),
]


def load_json(path: Path, hint: str) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError as e:
        raise SystemExit(f"FATAL: cannot read {path.name} ({hint}): {e}")
    except ValueError as e:
        raise SystemExit(f"FATAL: {path.name} is not valid JSON: {e}")


def declared(workflow_dir: Path, quantized: set, registry: dict) -> set:
    """Every managed (quant-swappable) basename the copied workflows claim
    to load, from widgets and properties.models, top level and subgraphs."""
    names = set()
    for wf in workflow_dir.rglob("*.json"):
        doc = json.loads(wf.read_text())
        groups = [doc.get("nodes", [])] + [
            sg.get("nodes", [])
            for sg in doc.get("definitions", {}).get("subgraphs", [])
        ]
        for group in groups:
            for n in group:
                for v in n.get("widgets_values") or []:
                    if isinstance(v, str) and v in quantized:
                        names.add(v)
                for m in (n.get("properties") or {}).get("models") or []:
                    if m.get("name") in quantized:
                        names.add(m["name"])
                        assert m.get("url") == registry[m["name"]]["url"], (
                            f"{wf.name}: {m['name']} declares url "
                            f"{m.get('url')}, registry says "
                            f"{registry[m['name']]['url']}"
                        )
    return names


def normalized_links(raw_links: list) -> list:
    """Normalize root array links and subgraph object links."""
    result = []
    for link in raw_links or []:
        if isinstance(link, list):
            assert len(link) >= 5, f"malformed root link: {link}"
            result.append({
                "id": link[0], "origin_id": link[1],
                "origin_slot": link[2], "target_id": link[3],
                "target_slot": link[4],
            })
        else:
            result.append(link)
    return result


def assert_graph_integrity(graph: dict, label: str, root: bool = False) -> None:
    """Every serialized link must resolve through both endpoint slots."""
    nodes = graph.get("nodes") or []
    node_ids = [node["id"] for node in nodes]
    assert len(node_ids) == len(set(node_ids)), f"{label}: duplicate node id"
    by_id = {node["id"]: node for node in nodes}
    pseudo_ids = {
        node["id"] for node in (graph.get("inputNode"),
                                graph.get("outputNode"))
        if isinstance(node, dict)
    }

    links = normalized_links(graph.get("links") or [])
    link_ids = [link["id"] for link in links]
    assert len(link_ids) == len(set(link_ids)), f"{label}: duplicate link id"
    by_link_id = {link["id"]: link for link in links}

    for link in links:
        link_id = link["id"]
        origin_id = link["origin_id"]
        target_id = link["target_id"]
        assert origin_id in by_id or origin_id in pseudo_ids, (
            f"{label}: link {link_id} has missing origin {origin_id}"
        )
        assert target_id in by_id or target_id in pseudo_ids, (
            f"{label}: link {link_id} has missing target {target_id}"
        )
        if origin_id in by_id:
            outputs = by_id[origin_id].get("outputs") or []
            slot = link["origin_slot"]
            assert isinstance(slot, int) and slot < len(outputs), (
                f"{label}: link {link_id} has invalid origin slot {slot}"
            )
            assert link_id in (outputs[slot].get("links") or []), (
                f"{label}: origin {origin_id}:{slot} omits link {link_id}"
            )
        if target_id in by_id:
            inputs = by_id[target_id].get("inputs") or []
            slot = link["target_slot"]
            assert isinstance(slot, int) and slot < len(inputs), (
                f"{label}: link {link_id} has invalid target slot {slot}"
            )
            assert inputs[slot].get("link") == link_id, (
                f"{label}: target {target_id}:{slot} omits link {link_id}"
            )

    for node in nodes:
        for output in node.get("outputs") or []:
            for link_id in output.get("links") or []:
                assert link_id in by_link_id, (
                    f"{label}: node {node['id']} output references missing "
                    f"link {link_id}"
                )
        for input_slot in node.get("inputs") or []:
            link_id = input_slot.get("link")
            assert link_id is None or link_id in by_link_id, (
                f"{label}: node {node['id']} input references missing "
                f"link {link_id}"
            )

    if root and node_ids:
        assert graph["last_node_id"] >= max(node_ids), (
            f"{label}: last_node_id is below a live node id"
        )
    if root and link_ids:
        assert graph["last_link_id"] >= max(link_ids), (
            f"{label}: last_link_id is below a live link id"
        )


def assert_source_workflows(registry: dict) -> None:
    workflow_root = REPO / "workflows" / "MiniMax H3"
    workflow_files = sorted(workflow_root.rglob("*.json"))
    expected_workflows = set(SOURCE_WORKFLOW_DEFAULTS) | {
        f"Upscaling/{name}" for name in UPSCALING_WORKFLOWS
    } | REFMOD_WORKFLOWS
    found_workflows = {
        workflow_file.relative_to(workflow_root).as_posix()
        for workflow_file in workflow_files
    }
    assert found_workflows == expected_workflows, (
        "unexpected MiniMax workflow set: "
        f"missing={sorted(expected_workflows - found_workflows)}, "
        f"extra={sorted(found_workflows - expected_workflows)}"
    )

    for workflow_file in workflow_files:
        doc = load_json(workflow_file, "workflow")
        label = workflow_file.relative_to(workflow_root).as_posix()
        assert_graph_integrity(doc, label, root=True)
        for subgraph in doc.get("definitions", {}).get("subgraphs", []):
            assert_graph_integrity(
                subgraph,
                f"{label}/{subgraph.get('name', subgraph['id'])}",
            )

    for name in sorted(UPSCALING_WORKFLOWS):
        doc = load_json(workflow_root / "Upscaling" / name,
                        "upscaling workflow")
        groups = [doc.get("nodes", [])] + [
            subgraph.get("nodes", [])
            for subgraph in doc.get("definitions", {}).get("subgraphs", [])
        ]
        nodes = [node for group in groups for node in group]
        upscalers = [
            node for node in nodes
            if node.get("type") == "MinimaxH3LatentUpscaler3D"
        ]
        assert len(upscalers) == 1, (
            f"{name}: expected one MinimaxH3LatentUpscaler3D node"
        )
        upscaler = upscalers[0]
        assert upscaler.get("widgets_values") == [
            LATENT_UPSCALER, "scale by multiplier", 2, 32, False, False,
            "cuda", "fp16",
        ], f"{name}: latent upscaler settings drifted"
        assert upscaler.get("widgets_values_named") == {
            "model_name": LATENT_UPSCALER,
            "mode": "scale by multiplier",
            "mode.scale": 2,
            "align": 32,
            "enable_temporal_chunking": False,
            "force_unload": False,
            "device": "cuda",
            "precision": "fp16",
        }, f"{name}: named latent upscaler settings drifted"

        power_lora_nodes = [
            node for node in nodes
            if node.get("type") == "Power Lora Loader (rgthree)"
        ]
        assert len(power_lora_nodes) == 1, (
            f"{name}: expected one character LoRA placeholder"
        )
        power_lora = power_lora_nodes[0]
        lora_widget = power_lora["widgets_values"][2]
        named_lora_widget = power_lora["widgets_values_named"]["lora_1"]
        expected_lora_widget = {
            "on": False,
            "lora": CHARACTER_LORA_PLACEHOLDER,
            "strength": 1,
            "strengthTwo": None,
        }
        assert lora_widget == expected_lora_widget, (
            f"{name}: character LoRA widget is not the safe placeholder"
        )
        assert named_lora_widget == expected_lora_widget, (
            f"{name}: named character LoRA widget is not the safe placeholder"
        )

    for name, defaults in SOURCE_WORKFLOW_DEFAULTS.items():
        diffusion, lora, strength, has_input_image = defaults
        doc = load_json(workflow_root / name, "workflow defaults")
        loaders = [node for node in doc["nodes"]
                   if node.get("type") == "UNETLoader"]
        assert len(loaders) == 1, f"{name}: expected one UNETLoader"
        loader = loaders[0]
        assert loader.get("widgets_values") == [diffusion, "default"], (
            f"{name}: wrong default diffusion model: "
            f"{loader.get('widgets_values')}"
        )
        assert (loader.get("widgets_values_named") or {}).get(
            "unet_name") == diffusion, f"{name}: named UNET value drifted"
        assert (loader.get("properties") or {}).get("models") == [{
            "name": diffusion,
            "url": registry[diffusion]["url"],
            "directory": registry[diffusion]["subdir"],
        }], f"{name}: UNET properties.models does not match the registry"

        lora_nodes = [node for node in doc["nodes"]
                      if node.get("type") == "LoraLoaderModelOnly"]
        assert len(lora_nodes) == 1, f"{name}: expected one Turbo LoRA loader"
        lora_node = lora_nodes[0]
        assert lora_node.get("widgets_values") == [lora, strength], (
            f"{name}: Turbo LoRA does not match its 8-step schedule"
        )
        assert lora_node.get("widgets_values_named") == {
            "lora_name": lora, "strength_model": strength,
        }, f"{name}: named Turbo LoRA values drifted"

        schedulers = [node for node in doc["nodes"]
                      if node.get("type") == "BasicScheduler"]
        assert len(schedulers) == 1, f"{name}: expected one BasicScheduler"
        assert schedulers[0].get("widgets_values") == ["simple", 8, 1], (
            f"{name}: expected the tuned 8-step simple schedule"
        )
        assert not any(node.get("type") == "BetaSamplingScheduler"
                       for node in doc["nodes"]), (
            f"{name}: obsolete BetaSamplingScheduler remains"
        )

        if has_input_image:
            image_nodes = [node for node in doc["nodes"]
                           if node.get("type") == "LoadImage"]
            assert len(image_nodes) == 1, f"{name}: expected one LoadImage"
            assert image_nodes[0].get("widgets_values") == [
                "your_input_image.png", "image"
            ], f"{name}: stale local image default remains"

    print("✅ all fifteen workflow graphs have internally consistent links")
    print("✅ five refreshed workflows use the int8 defaults and 8-step Turbo")
    print("✅ five upscaling workflows use the pinned 2x latent upscaler")

    cached = load_json(REPO / "workflows" / CACHED_WORKFLOW, "cached workflow")
    unets = [node for node in cached["nodes"] if node.get("type") == "UNETLoader"]
    clips = [node for node in cached["nodes"] if node.get("type") == "CLIPLoader"]
    assert len(unets) == 1 and len(clips) == 1
    assert unets[0]["widgets_values"][0] == REF2VA_INT8
    assert unets[0]["properties"]["models"][0]["name"] == REF2VA_INT8
    assert clips[0]["widgets_values"][0] == TEXT_ENCODER
    assert clips[0]["properties"]["models"][0]["name"] == TEXT_ENCODER
    cached_text = (REPO / "workflows" / CACHED_WORKFLOW).read_text()
    assert "minimax_h3_fl2va_" not in cached_text
    assert "nvfp4_awq" not in cached_text
    assert IGNORED_LORA in cached_text
    print("✅ cached reference-to-video workflow loads Ref2VA and the int8 encoder")


def main() -> int:
    template = load_json(REPO / "template.json",
                         "written by the migration's slice A; this test only "
                         "goes green once the slices are integrated")
    registry = load_json(REPO / "src" / "models_registry.json", "registry")

    assert_source_workflows(registry)

    groups = template.get("swap_groups") or []
    assert len(groups) == 1, f"expected exactly one swap group, got {len(groups)}"
    group = groups[0]
    assert group["env"] == "minimax_quant", group["env"]
    assert group["default"] == "int8", group["default"]
    profiles = group["profiles"]
    assert set(profiles) == {"int8", "fp8", "nvfp4", "false", "bf16"}, profiles
    for profile in profiles.values():
        assert set(profile) == {"ref2va", "text_encoder"}, profile
    quantized = {
        p[role] for p in profiles.values()
        for role in ("ref2va", "text_encoder")
    }
    diffusion_models = {
        profile["ref2va"] for profile in profiles.values()
    }

    assert {role: profiles["false"][role] for role in BF16_MODELS} == (
        BF16_MODELS
    ), profiles["false"]
    for role, basename in BF16_MODELS.items():
        assert registry[basename]["subdir"] == "diffusion_models", (
            f"{role}: {basename} must install under diffusion_models"
        )
        assert registry[basename]["url"] == (
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
            f"diffusion_models/{basename}"
        ), f"{role}: unexpected bf16 URL {registry[basename]['url']}"
    print("✅ bf16 profile points at the Ref2VA Comfy-Org model")

    # The DiT quant varies by card. The text encoder does not: every profile
    # ships Comfy-Org's stock int8 build, so no quant can pull a second
    # encoder onto the volume.
    for quant, profile in sorted(profiles.items()):
        assert profile["text_encoder"] == TEXT_ENCODER, (
            f"{quant}: text_encoder is {profile['text_encoder']}, "
            f"expected {TEXT_ENCODER}"
        )
    strays = [b for b in registry
              if b.startswith("qwen3vl") and b != TEXT_ENCODER]
    assert not strays, f"registry carries unused text encoders: {sorted(strays)}"
    print(f"✅ every model profile loads {TEXT_ENCODER}")

    for quant, profile in profiles.items():
        assert "hyperflow" not in profile and "hyperflow_node" not in profile, (
            f"{quant}: HyperFlow weights must not be queued for this workflow"
        )
    for basename in (HYPERFLOW_FULL, HYPERFLOW_PRUNED):
        assert registry[basename] == {
            "url": HYPERFLOW_URL_BASE + basename,
            "subdir": "loras",
            "min_size_mb": 3700,
        }, f"unexpected HyperFlow registry entry: {basename}"
    assert registry[HYPERFLOW_NODE_WEIGHT] == {
        "url": HYPERFLOW_NODE_URL,
        "subdir": "hyperflow",
        "min_size_mb": 3700,
    }, "node-specific HyperFlow weight must go in models/hyperflow"
    assert "hyperflow_node" not in {
        role for profile in profiles.values() for role in profile
    }, "node-specific HyperFlow weight must not be a selected profile file"
    assert all(
        basename not in workflow.read_text()
        for workflow in (REPO / "workflows").rglob("*.json")
        for basename in (HYPERFLOW_FULL, HYPERFLOW_PRUNED, HYPERFLOW_NODE_WEIGHT)
    ), "ghost HyperFlow model must not appear in shipped workflows"

    for b in TURBO_LORAS:
        assert b in registry, f"turbo LoRA missing from registry: {b}"
        assert registry[b]["subdir"] == "loras", (
            f"{b}: subdir is {registry[b]['subdir']}, expected loras"
        )
    for b in BUNDLED_LORAS:
        assert registry[b]["url"] == (
            "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/"
            f"{b}"
        ), f"{b}: unexpected LightX2V URL {registry[b]['url']}"
    flag = template["flags"]["download_minimax_h3"]
    removed_from_registry = REMOVED_TURBO_LORAS & set(registry)
    assert not removed_from_registry, (
        f"retired Turbo LoRAs remain registered: {sorted(removed_from_registry)}"
    )
    removed_from_extras = REMOVED_TURBO_LORAS & set(
        flag.get("extra_models", [])
    )
    assert not removed_from_extras, (
        f"retired Turbo LoRAs remain bundled: {sorted(removed_from_extras)}"
    )
    assert LATENT_UPSCALER in registry, "latent upscaler missing from registry"
    assert registry[LATENT_UPSCALER] == {
        "url": LATENT_UPSCALER_URL,
        "subdir": "latent_upscale_models",
        "min_size_mb": 650,
    }, f"unexpected latent upscaler registry entry: {registry[LATENT_UPSCALER]}"
    assert flag.get("workflows") == [CACHED_WORKFLOW], flag.get("workflows")
    assert not flag.get("folders"), flag.get("folders")
    assert not flag.get("extra_models"), flag.get("extra_models")
    assert template.get("auto_download") == [IGNORED_LORA], (
        template.get("auto_download")
    )
    print("✅ download_minimax_h3 provisions only the cached reference workflow")

    custom_nodes = template["custom_nodes"]
    assert custom_nodes["target"] == "image", custom_nodes
    from test_viggle_provisioner import VIGGLE_NODE
    assert custom_nodes.get("repos") == [LATENT_UPSCALER_NODE, REFMOD_NODE, VIGGLE_NODE], (
        f"template custom-node list must preserve exact pins, got {custom_nodes}"
    )
    assert custom_nodes.get("profile_repos") == {"minimax_quant": {
        "false": [HYPERFLOW_NODE], "bf16": [HYPERFLOW_NODE]}}, (
        "HyperFlow node must only be selected for full BF16 profiles")
    assert "hyperflow" in template.get("extra_model_paths", []), (
        "HyperFlow model folder must be visible to ComfyUI")
    print("✅ latent upscaler model destination and custom-node pin are exact")

    assert "refmods" in template.get("extra_model_paths", []), "RefMods must persist on the volume"

    # Run the real hook against an isolated filesystem. Only absolute path
    # probes and mkdir are redirected; branch selection remains in the hook.
    hook_driver = r"""
        test() {
            if [[ "$1" == "-d" && "$2" == "/workspace" ]]; then
                builtin test -d "$TEST_FS/workspace"
            else
                builtin test "$@"
            fi
        }
        mkdir() {
            local args=() arg
            for arg in "$@"; do
                if [[ "$arg" == /* ]]; then
                    args+=("$TEST_FS$arg")
                else
                    args+=("$arg")
                fi
            done
            command mkdir "${args[@]}"
        }
        source "$1"
    """
    for workspace_exists in (False, True):
        with tempfile.TemporaryDirectory() as hook_tmp:
            fs = Path(hook_tmp) / "test filesystem"
            fs.mkdir()
            if workspace_exists:
                (fs / "workspace").mkdir()
            root = fs / ("workspace/ComfyUI" if workspace_exists else "ComfyUI")
            env = dict(os.environ, TEST_FS=str(fs))
            env.pop("PERSIST_ROOT", None)
            for iteration in range(2):
                subprocess.run(["bash", "-c", hook_driver, "hook",
                                str(REPO / "src/hooks/pre_launch.sh")],
                               env=env, check=True)
                assert (root / "input/refmod_images").is_dir()
                assert (root / "input/refmod_video").is_dir()
                marker = root / "input/refmod_images/keep.txt"
                if iteration:
                    assert marker.read_text() == "user reference"
                marker.write_text("user reference")
            other = fs / ("ComfyUI" if workspace_exists else "workspace")
            assert not other.exists(), "hook wrote into the wrong root"
    print("✅ RefMod input folders exist with/without workspace and survive repeated boots")

    provisioner = runtime_dir() / "src" / "provisioner.py"
    assert provisioner.is_file(), f"no provisioner at {provisioner}"
    from test_viggle_provisioner import check as check_viggle
    check_viggle(provisioner)

    manifests: dict = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for label, raw_quant, key, warning in CASES:
            slug = re.sub(r"[^A-Za-z0-9]", "_", label)
            dst = tmp / f"wf-{slug}"
            manifest = tmp / f"manifest-{slug}.tsv"
            env = dict(os.environ)
            env["download_minimax_h3"] = "true"
            env.pop("minimax_quant", None)
            if raw_quant is not None:
                env["minimax_quant"] = raw_quant
            proc = subprocess.run(
                [sys.executable, str(provisioner),
                 "--template", str(REPO / "template.json"),
                 "--registry", str(REPO / "src" / "models_registry.json"),
                 "--workflows-src", str(REPO / "workflows"),
                 "--workflows-dst", str(dst),
                 "--models-root", str(tmp / f"models-{slug}"),
                 "--manifest", str(manifest)],
                env=env, capture_output=True, text=True,
            )
            assert proc.returncode == 0, (
                f"{label}: provisioner exited {proc.returncode}\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
            lines = [l for l in manifest.read_text().splitlines() if l]
            copied = sorted(
                path.relative_to(dst).as_posix()
                for path in dst.rglob("*.json")
            )
            assert copied == [CACHED_WORKFLOW], (
                f"{label}: copied {copied}, expected only {CACHED_WORKFLOW}"
            )
            downloaded = {l.split("\t")[1].rsplit("/", 1)[1] for l in lines}
            manifests[label] = {l.split("\t", 1)[0] for l in lines}

            wanted = {
                profiles[key]["ref2va"],
                profiles[key]["text_encoder"],
            }
            got = declared(dst, quantized, registry)
            assert got == wanted, (
                f"{label}: workflows declare {sorted(got)}, "
                f"selected profile {key!r} is {sorted(wanted)}"
            )
            expected = wanted | CACHED_STATIC_MODELS
            assert downloaded == expected, (
                f"{label}: manifest is {sorted(downloaded)}, "
                f"expected {sorted(expected)}"
            )
            assert downloaded & diffusion_models == {profiles[key]["ref2va"]}
            assert IGNORED_LORA not in downloaded, (
                f"{label}: ignored NSFW LoRA was queued"
            )
            skipped_families = {
                HYPERFLOW_FULL, HYPERFLOW_PRUNED, HYPERFLOW_NODE_WEIGHT,
                LATENT_UPSCALER, *BUNDLED_LORAS,
            } - {REF2VA_TURBO_8STEP}
            assert not downloaded & skipped_families, (
                f"{label}: queued models this workflow does not load: "
                f"{sorted(downloaded & skipped_families)}"
            )
            if warning:
                assert warning in proc.stdout, (
                    f"{label}: expected warning {warning!r}, got:\n{proc.stdout}"
                )
            assert IGNORED_LORA not in proc.stdout, (
                f"{label}: ignored NSFW LoRA was reported as user-supplied:\n"
                f"{proc.stdout}"
            )
            print(f"✅ {label} -> {key}: only the cached workflow and its "
                  f"{len(expected)} models were queued")

    # Fallback and aliases must be byte-identical in URL terms, not merely
    # "some default-ish" sets.
    assert (manifests["invalid-quant"] == manifests["int8"] ==
            manifests["unset"]), (
        "unset / int8 / invalid quant must queue the same URLs"
    )
    assert manifests["fp8"] == manifests["FP8"], (
        "quant matching must ignore case"
    )
    assert (manifests["false"] == manifests["FALSE"] ==
            manifests["false-whitespace"]), (
        "false must select bf16 regardless of case or surrounding whitespace"
    )
    assert manifests["bf16"] == manifests["BF16"] == manifests["false"], (
        "bf16 and false must queue identical model URLs")
    subprocess.run([sys.executable, str(REPO / "tools" / "test_auto_prompt.py")], check=True)
    print("✅ all minimax_quant profiles consistent; fallbacks are safe")
    return 0


if __name__ == "__main__":
    sys.exit(main())

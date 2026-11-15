import json, sys
print(json.dumps({
    "cpu": {"name": "Fake CPU", "logical_cores": 8, "physical_cores": 4},
    "memory": {"total": 32e9, "available": 20e9},
    "disk": {"free_bytes": 500e9, "warnings": []},
    "gpu": {"vulkan": None, "nvidia": [], "has_gpu": False},
}))

import sys
print("Start", flush=True)
try:
    from kbprojection.loaders.sick import SICKLoader
    print("Imported SICK", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
print("Done", flush=True)

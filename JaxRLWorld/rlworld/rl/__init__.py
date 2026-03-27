try:
    import genesis as gs
    if not gs._initialized:
        gs.init(logging_level="warning")
except ImportError:
    pass

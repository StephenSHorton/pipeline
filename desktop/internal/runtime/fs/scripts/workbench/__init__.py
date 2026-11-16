"""Agent workbench: Claude makes, measures and fixes assets in the pipeline.

clay.py holds SDF blobs (Clide3D's field names, so scenes can converge with that project later) and
mesher.py turns them into a textured GLB that the pipeline's retexture/psx stages accept. eyes.py
looks at a mesh, measures it and compares it with its reference photo. Model-loading work stays in
the factory; everything here is CPU only.
"""

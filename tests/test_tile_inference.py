import numpy as np
from tile_inference import Detection, TileGenerator, clip_and_translate, global_nms

def test_full_coverage_and_boundary_tile():
    image=np.zeros((1000,1500,3),np.uint8); tiles=TileGenerator(640,640,.15).generate(image)
    assert min(t.x for t in tiles)==0 and max(t.x+t.image.shape[1] for t in tiles)==1500
    assert min(t.y for t in tiles)==0 and max(t.y+t.image.shape[0] for t in tiles)==1000

def test_small_image_is_one_undistorted_tile():
    image=np.zeros((120,300,3),np.uint8); tile=TileGenerator().generate(image)[0]
    assert (tile.x,tile.y,tile.image.shape[:2])==(0,0,(120,300))

def test_coordinate_transform_and_clip():
    translated=clip_and_translate(np.array([[100,80,300,250],[-10,-10,1000,1000]]),800,400,1200,700)
    assert np.allclose(translated[0],[900,480,1100,650])
    assert np.allclose(translated[1],[790,390,1200,700])

def test_global_nms_removes_overlapping_duplicate():
    detections=[Detection(np.array([900,480,1100,650]),.91,0),Detection(np.array([902,482,1102,652]),.87,0),Detection(np.array([902,482,1102,652]),.87,1)]
    kept=global_nms(detections,.5)
    assert len(kept)==2 and any(d.class_id==1 for d in kept)


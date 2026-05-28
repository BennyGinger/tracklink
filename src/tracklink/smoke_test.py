
if __name__ == "__main__":

    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc
    from pathlib import Path
    from fits_io import FitsIO

    folder = Path("/media/ben/Analysis/Python/Images/zymosan/zym_chamber_500k_WT_HoxB8_001_s1")
    img_path = folder / "fits_array.tif"
    img_reader = FitsIO.from_path(img_path)
    
    mask_path = folder / "fits_mask.tif"
    mask_reader = FitsIO.from_path(mask_path)
    
    img_gfp = img_reader.get_channel_array('BFP')
    if isinstance(img_gfp, list):
        raise ValueError("Expected a single array for the BFP channel, but got a list.")
    
    mask_gfp = mask_reader.get_array()
    if isinstance(mask_gfp, list):
        raise ValueError("Expected a single array for the BFP mask channel, but got a list.")
    
    model = Trackastra.from_pretrained("general_2d")

    track_graph, masks_tracked = model.track(
        img_gfp,
        mask_gfp,
        mode="greedy_nodiv"
    )
    
    ctc_tracks, ctc_masks = graph_to_ctc(
    track_graph,
    masks_tracked,
)
    
    mask_reader.save_array(ctc_masks, "TYX", "GFP", "test_track.tif", )
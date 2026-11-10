
if __name__ == "__main__":

    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc
    from pathlib import Path
    from scope_io import ScopeIO
    import pandas as pd
    import numpy as np
    from skimage.segmentation import relabel_sequential

    folder = Path("/media/ben/Analysis/Python/Images/zymosan/zym_chamber_500k_WT_HoxB8_001_s1")
    img_path = folder / "fits_array.tif"
    img_reader = ScopeIO.from_path(img_path)
    
    mask_path = folder / "fits_mask.tif"
    mask_reader = ScopeIO.from_path(mask_path)
    
    img_gfp = img_reader.get_channel('BFP').array
    
    mask_gfp = mask_reader.get_array().array
    
    model = Trackastra.from_pretrained("general_2d")

    
    track_graph, masks_tracked = model.track(
        img_gfp,
        mask_gfp,
        mode="greedy_nodiv",
    )
    
    ctc_tracks, ctc_masks = graph_to_ctc(track_graph, masks_tracked,)
    ctc_masks = np.asarray(ctc_masks)
    ctc_tracks['length'] = ctc_tracks['t2'] - ctc_tracks['t1']
    
    label_to_remove = ctc_tracks.loc[ctc_tracks['length'] < 120, 'label'].tolist()
    
    # Remove tracks with length less than 120 frames
    filtered_masks = ctc_masks.copy()
    filtered_masks[np.isin(filtered_masks, label_to_remove)] = 0

    csv_path = folder / "test_track.csv"
    ctc_tracks.to_csv(csv_path, index=False)
    mask_reader.save_array(ctc_masks, channel_labels="GFP", output_name="test_track.tif")
    mask_reader.save_array(filtered_masks, channel_labels="GFP", output_name="test_track_filtered.tif")

from pathlib import Path

from cellpose import io, models, train

def main(dataset_dir: Path):
    # -----------------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------------

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"


    # -----------------------------------------------------------------------------
    # Load training / test data
    # -----------------------------------------------------------------------------

    output = io.load_train_test_data(
        str(train_dir),
        str(test_dir),
        image_filter=None,
        mask_filter="_seg.npy",
    )

    train_images, train_labels, train_image_names, \
    test_images, test_labels, test_image_names = output


    # -----------------------------------------------------------------------------
    # Initialize pretrained Cellpose model
    # -----------------------------------------------------------------------------

    model = models.CellposeModel(gpu=True)


    # -----------------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------------

    model_path, train_loss, test_loss= train.train_seg(
        model.net,
        train_data=train_images,
        train_labels=train_labels,
        train_files=train_image_names,

        test_data=test_images,
        test_labels=test_labels,
        test_files=test_image_names,

        normalize=True,

        learning_rate=1e-5,
        weight_decay=0.1,

        n_epochs=100,
        batch_size=1,

        model_name="neutro_enhanced_cpsam",
        min_train_masks=1,

        save_path=str(dataset_dir),
    )


    print(f"\nModel saved to:\n{model_path}")
    print(f"\nFinal training loss: {train_loss[-1]:.4f}")
    print(f"Final test loss: {test_loss[-1]:.4f}")




if __name__ == "__main__":
    import numpy as np
    from pathlib import Path

    dataset_dir = Path("/media/ben/Analysis/Python/Cellpose_models/Neutrophils_Enhanced/")

    main(dataset_dir)
    
    # def check_cellpose_pairs(folder):
    #     folder = Path(folder)

    #     images = {
    #         p.stem: p
    #         for p in folder.glob("*")
    #         if p.suffix.lower() in [".tif", ".tiff", ".png", ".jpg"]
    #         and not p.name.endswith("_seg.npy")
    #     }

    #     segs = {
    #         p.name.replace("_seg.npy", ""): p
    #         for p in folder.glob("*_seg.npy")
    #     }

    #     missing_seg = sorted(set(images) - set(segs))
    #     missing_img = sorted(set(segs) - set(images))

    #     print(f"\nFolder: {folder}")
    #     print(f"Images: {len(images)}")
    #     print(f"Seg files: {len(segs)}")

    #     if missing_seg:
    #         print("\nImages without _seg.npy:")
    #         for name in missing_seg:
    #             print(" ", images[name].name)

    #     if missing_img:
    #         print("\n_seg.npy without image:")
    #         for name in missing_img:
    #             print(" ", segs[name].name)

    #     if not missing_seg and not missing_img:
    #         print("All image/_seg.npy pairs look matched.")


    # check_cellpose_pairs((dataset_dir / "train"))
    # check_cellpose_pairs((dataset_dir / "test"))
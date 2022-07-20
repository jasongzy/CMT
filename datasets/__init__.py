from datasets import kitti, nuscenes_data, sampler, waymo_data


def get_dataset(config, type="train", **kwargs):
    if config.dataset == "kitti":
        data = kitti.kittiDataset(
            path=config.path,
            split=kwargs.get("split", "train"),
            category_name=config.category_name,
            coordinate_mode=config.coordinate_mode,
            preloading=config.preloading,
            preload_offset=config.preload_offset if type != "test" else -1,
        )
    elif config.dataset == "nuscenes":
        data = nuscenes_data.NuScenesDataset(
            path=config.path,
            split=kwargs.get("split", "train_track"),
            category_name=config.category_name,
            version=config.version,
            key_frame_only=config.key_frame_only,
            preload_offset=config.preload_offset if type != "test" else -1,
            preloading=config.preloading,
            min_points=config.min_points
            if kwargs.get("split", "train_track") in [config.val_split, config.test_split]
            else -1,
        )
    elif config.dataset == "waymo":
        data = waymo_data.WaymoDataset(
            path=config.path,
            split=kwargs.get("split", "train_track"),
            category_name=config.category_name,
            preload_offset=config.preload_offset,
            preloading=config.preloading,
            tiny=config.tiny,
        )
    else:
        data = None

    if type == "train":
        return sampler.PointTrackingSampler(
            dataset=data, random_sample=config.random_sample, sample_per_epoch=config.sample_per_epoch, config=config
        )
    else:
        return sampler.TestTrackingSampler(dataset=data, config=config)

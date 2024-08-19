import argparse
import torch
from nerfstudio.data.scene_box import SceneBox
import os
from pathlib import Path
import sys
import yaml
from nerfstudio.configs import base_config as cfg
from rich.console import Console
from mesh_utils import extract_mesh
import numpy as np
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.io import load_from_json
from nerfstudio.data.dataparsers.tnt_dataparser import TNTParserConfig

CONSOLE = Console(width=120)

def load_model(config: cfg.Config, device):
    config.trainer.load_dir = config.get_checkpoint_dir()
    assert config.trainer.load_dir is not None
    if config.trainer.load_step is None:
        CONSOLE.print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.trainer.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"No checkpoint directory found at {config.trainer.load_dir}, ", justify="center")
            CONSOLE.print(
                "Please make sure the checkpoint exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.trainer.load_dir))[-1]
    else:
        load_step = config.trainer.load_step
    load_path = config.trainer.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    CONSOLE.print(f"Found {load_path}")
    loaded_state = torch.load(load_path, map_location="cpu")

    model_param = {}
    for key, item in loaded_state['pipeline'].items():
        if key.startswith('_model'):
            model_param[key.split('_model.')[1]] = item

    scene_box = SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32))
    if 'field.embedding_appearance.embedding.weight' in model_param:
        num_train_data = model_param['field.embedding_appearance.embedding.weight'].shape[0]
    else:
        num_train_data = 360        # random

    model = config.pipeline.model.setup(
        scene_box=scene_box,
        num_train_data=num_train_data,
        world_size=0,
        local_rank=0,
    )
    model.to(device)
    model.load_state_dict(model_param)

    return model


if __name__ == '__main__':
    CONSOLE = Console(width=120)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='outputs/hydoor-tnt-scan3-initbound-schevar/hydoor/2024-04-25_201904/config.yml')
    parser.add_argument('--output_path', type=str)
    parser.add_argument("--resolution", default=512, type=int, help="Marching cubes resolution")
    parser.add_argument("--suffix", default='', type=str, help="PLY file suffix")
    parser.add_argument("--block_res", default=128, type=int, help="Block-wise resolution for marching cubes")
    parser.add_argument("--keep_lcc", action="store_true",
                        help="Keep only largest connected component. May remove thin structures.")
    parser.add_argument("--vis_crop", action="store_true")

    args = parser.parse_args()

    config_path = Path(args.conf)
    if args.output_path is None:
        if args.suffix != '':
            output_path = Path('meshes/' + args.conf.split('/')[-4] + f'-{args.suffix}.ply')
        else:
            output_path = Path('meshes/' + args.conf.split('/')[-4] + '.ply')
    else:
        output_path = Path(args.output_path)

    torch.set_float32_matmul_precision('high')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, cfg.Config)

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.pipeline.datamanager.eval_image_indices = None

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.vis_crop:
        _, pipeline, _ = eval_setup(config_path)
        pipeline.eval()
        model = pipeline.model
    else:
        model = load_model(config, device)
        model.eval()



    sdf_func = lambda x: -model.field.forward_geonetwork(x)[:, 0].contiguous()
    bounds = np.array([[-1., 1.], [-1, 1.], [-1., 1.]])

    '''
    To ensure a fair comparison with Neuralangelo, we post-process the results to perform marching cubes within the same bounding box and at the same resolution as Neuralangelo.
    '''
    if isinstance(config.pipeline.datamanager.dataparser, TNTParserConfig):             
        print('Postprocess for tnt...')
        meta = load_from_json(config.pipeline.datamanager.dataparser.data / config.pipeline.datamanager.dataparser.scene_name / "transforms.json")

        if config.pipeline.datamanager.dataparser.scene_name != 'Meetingroom':  # we haven't scaled the indoor scene.
            print('Postprocess because of scale_poses...')
            frames = meta['frames']

            def _gl_to_cv(gl):
                # convert to CV convention used in Imaginaire
                cv = gl * torch.tensor([1, -1, -1, 1])
                return cv

            def get_camera(idx):
                # Camera intrinsics.
                intr = torch.tensor([[meta["fl_x"], meta["sk_x"], meta["cx"]],
                                    [meta["sk_y"], meta["fl_y"], meta["cy"]],
                                    [0, 0, 1]]).float()
                # Camera pose.
                c2w_gl = torch.tensor(frames[idx]["transform_matrix"], dtype=torch.float32)
                c2w = _gl_to_cv(c2w_gl)
                # center scene
                center = torch.tensor(meta["sphere_center"])
                c2w[:3, -1] -= center
                # scale scene
                scale = torch.tensor(meta["sphere_radius"])
                c2w[:3, -1] /= scale
                return intr, c2w

            camera_to_worlds = []
            for i in range(len(frames)):
                intrinsic, camtoworld = get_camera(i)
                camera_to_worlds.append(camtoworld)

            camera_to_worlds = torch.stack(camera_to_worlds)

            # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
            camera_to_worlds[:, 0:3, 1:3] *= -1

            scale_factor = float(torch.max(torch.abs(camera_to_worlds[:, :3, 3])))       ## Yifan: for single object
            scale_factor /= config.pipeline.datamanager.dataparser.scale_factor
            print('scale factor:', scale_factor)

            if "aabb_range" in meta:
                bounds = (np.array(meta["aabb_range"]) - np.array(meta["sphere_center"])[..., None]) / meta["sphere_radius"]
                bounds /= scale_factor
            else:
                bounds = np.array([[-1.2, 1.2], [-1.2, 1.2], [-1.2, 1.2]])
            print('bounds:', bounds)
            mesh = extract_mesh(sdf_func=sdf_func, bounds=bounds, intv=(2.0 / int(args.resolution*scale_factor)),
                                    block_res=args.block_res, texture_func=None, filter_lcc=args.keep_lcc)
            mesh.vertices = mesh.vertices * meta["sphere_radius"] * scale_factor + np.array(meta["sphere_center"])
        else:
            if "aabb_range" in meta:
                bounds = (np.array(meta["aabb_range"]) - np.array(meta["sphere_center"])[..., None]) / meta["sphere_radius"]
            else:
                bounds = np.array([[-1.2, 1.2], [-1.2, 1.2], [-1.2, 1.2]])
            mesh = extract_mesh(sdf_func=sdf_func, bounds=bounds, intv=(2.0 / args.resolution),
                        block_res=args.block_res, texture_func=None, filter_lcc=args.keep_lcc)
            mesh.vertices = mesh.vertices * meta["sphere_radius"] + np.array(meta["sphere_center"])
    else:
        mesh = extract_mesh(sdf_func=sdf_func, bounds=bounds, intv=(2.0 / args.resolution),
                            block_res=args.block_res, texture_func=None, filter_lcc=args.keep_lcc)

    print(f"vertices: {len(mesh.vertices)}")
    print(f"faces: {len(mesh.faces)}")
    mesh.update_faces(mesh.nondegenerate_faces())
    print(f'Save mesh to {output_path}')
    mesh.export(output_path)
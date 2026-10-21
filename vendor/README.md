# vendor/

Gitignored clones and apps. Do not commit them.

| path | what | how |
|---|---|---|
| `hunyuan3d-swift/` | MLX-Swift Hunyuan CLI (`hy3d`) | `scripts/bootstrap.sh` |
| `hunyuan3d-2/` | Official Tencent Hunyuan3D-2 + 2mv (MPS, `shape_mv.sh`) | optional |
| `mesh2motion-app/` | CC0 human clips + retargeter | `git clone --depth 1 https://github.com/Mesh2Motion/mesh2motion-app` |
| `AniGen-mac/` | MPS AniGen (people skins) | Mac-only |
| `autoremesher.app/` | quad remesher | Mac `.app` |

On a new machine run `scripts/bootstrap.sh draft`.

import torch
import torch.nn.functional as F


def tbn(triangles):
    a, b, c = triangles.unbind(-2)
    n = F.normalize(torch.cross(b - a, c - a), dim=-1)
    d = b - a

    X = F.normalize(torch.cross(d, n), dim=-1)
    Y = F.normalize(torch.cross(d, X), dim=-1)
    Z = F.normalize(d, dim=-1)

    return torch.stack([X, Y, Z], dim=3)


def triangle2projection(triangles):
    R = tbn(triangles)
    T = triangles.unbind(-2)[0]
    I = torch.repeat_interleave(torch.eye(4, device=triangles.device)[None, None, ...], R.shape[1], 1)

    I[:, :, 0:3, 0:3] = R
    I[:, :, 0:3, 3] = T

    return I


def project_position(pts, deformed_triangles, canonical_triangles, tryiii, mask):
    if tryiii == None:
        Rt_def = torch.inverse(triangle2projection(deformed_triangles))[0]
        Rt_canon = triangle2projection(canonical_triangles)[0]
        homo = torch.cat([pts, torch.ones_like(pts)[:, 0:1]], dim=1)[:, :, None]

        def_local = torch.matmul(Rt_def, homo)
        def_canon = torch.matmul(Rt_canon, def_local)
    else:
        def_local, _ = interpolate(pts, deformed_triangles, canonical_triangles, tryiii, mask)

        homo = torch.cat([pts, torch.ones_like(pts)[:, 0:1]], dim=1)[:, :, None]

        def_canon = torch.matmul(def_local, homo)

    return def_canon[:, 0:3, 0].float()


def project_direction(dirs, deformed_triangles, canonical_triangles, tryiii=None):
    if tryiii == None:
        Rt_def = torch.inverse(triangle2projection(deformed_triangles))[0][:, 0:3, 0:3]
        Rt_canon = triangle2projection(canonical_triangles)[0][:, 0:3, 0:3]

        def_local = torch.matmul(Rt_def, dirs[:, :, None])
        def_canon = torch.matmul(Rt_canon, def_local)

    return def_canon[:, 0:3, 0].float()


def interpolate(pts, tris, trisc, tryiii, mask):
    msk_tryi = tryiii != 0
    tryiii[tryiii == 0] = 1
    tt = tryiii[mask]  ########################
    tt1 = tt.reshape(-1, 1)[:, 0]  ########################
    v0, _, _ = tris.unbind(-2)
    msk_tryi_111 = ((msk_tryi[mask]).unsqueeze(-1)).unsqueeze(-1)
    cent_111 = (v0[0])[(tt1 - 1).long()].reshape(tt.shape[0], 19, 3)
    dist = torch.norm(cent_111 - pts.reshape(-1, 3).unsqueeze(-2).repeat((1, 19, 1)), dim=2, keepdim=True)  # 19
    wts = torch.exp(-26.05000 * dist).unsqueeze(-1)

    mmt = triangle2projection(tris)[0]
    mmtc = triangle2projection(trisc)[0]
    mmt = torch.inverse(mmt)
    def_local = torch.matmul(mmtc, mmt)
    def_local = (def_local[(tt1 - 1).long()].reshape(tt.shape[0], 19, 4, 4) * wts * msk_tryi_111).sum(dim=1) / (
        (wts * msk_tryi_111).sum(dim=1)
    )  # 19

    return def_local, 1 / torch.det(def_local)  # mmt2, mmt2c

from __future__ import annotations

import vector

vector.register_awkward()

# warnings.filterwarnings("error", category=torch.UserWarning)


def _p4_from_ptetaphim(pt, eta, phi, mass):
    """ """
    return vector.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass})

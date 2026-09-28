"""Phase portrait subproject — composable, native-Matplotlib phase portraits.

Exports the main entry point and all public classes.
"""

from .masks import Mask, R2Mask
from .layers import Layer, EquilibriaLayer, NullclineLayer, QuiverLayer, OrbitLayer
from .portrait import PhasePortrait
from .interactive import Interactor
from .gallery import gallery_local

__all__ = [
    'Mask', 'R2Mask',
    'Layer', 'EquilibriaLayer', 'NullclineLayer', 'QuiverLayer', 'OrbitLayer',
    'PhasePortrait',
    'Interactor',
    'gallery_local',
]
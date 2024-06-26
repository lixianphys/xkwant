try:
    import palettable

    DEFAULT_CMAP = palettable.scientific.sequential.Bilbao_20.mpl_colormap
except ModuleNotFoundError:
    DEFAULT_CMAP = "jet"
LATTICE_CONST_HGTE = 0.646  # lattice constant of HgTe in nm from Wikipedia

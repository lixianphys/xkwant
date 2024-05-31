try:
    import palettable

    DEFAULT_CMAP = palettable.scientific.sequential.Bilbao_20.mpl_colormap
except ModuleNotFoundError:
    DEFAULT_CMAP = "jet"

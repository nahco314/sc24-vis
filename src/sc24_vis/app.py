import typer

from sc24_vis.vis_overall import vis_overall
from sc24_vis.vis_zoom import vis_zoom

app = typer.Typer()
app.command("vis")(vis_overall)
app.command("vis-zoom")(vis_zoom)

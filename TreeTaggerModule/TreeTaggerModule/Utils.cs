using ArcGIS.Core.CIM;
using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Desktop.Mapping;
using System;
using System.Collections.Generic;
using System.Windows;

namespace TreeTaggerModule
{
    static class Utils
    {

        public static string AddinAssemblyLocation()
        {
            var asm = System.Reflection.Assembly.GetExecutingAssembly();

            return System.IO.Path.GetDirectoryName(Uri.UnescapeDataString(new Uri(asm.Location).LocalPath));
        }

        public static string GetLayerPath(Layer layer)
        {
            return Uri.UnescapeDataString(layer.GetPath().LocalPath);
        }

        public static (List<string>, List<double[]>, List<double[]>, List<double>) GetRasterData(List<RasterLayer> rLayers)
        {
            var filePaths = new List<string>();
            var coords = new List<double[]>();
            var scales = new List<double>();
            var sizes = new List<double[]>();

            foreach (var raster in rLayers)
            {
                string path = GetLayerPath(raster);

                if (!string.IsNullOrEmpty(path) && !filePaths.Contains(path))
                {
                    filePaths.Add(path);

                    var extend = raster.GetRaster().GetExtent();

                    coords.Add([extend.XMin, extend.YMax]);

                    sizes.Add([raster.GetRaster().GetWidth(), raster.GetRaster().GetHeight()]);

                    scales.Add(raster.GetRaster().GetMeanCellSize().Item1);

                }
            }

            return (filePaths, coords, sizes, scales);
        }

        public async static void LoadShapeFiles(bool group=false)
        {
            var map = MapView.Active.Map;

            //get the newly create shape files containing the lines and polygon regions and add them to the current arcgis project
            await QueuedTask.Run(() =>
            {
                //getting directory paths
                var pathProject = ArcGIS.Desktop.Core.Project.Current.URI;
                string shapeFilesPath = System.IO.Path.GetDirectoryName(pathProject) + "\\TreeTagger";

                List<string> subDirectories = [.. System.IO.Directory.GetDirectories(shapeFilesPath)];
                subDirectories.RemoveAll(x => !x.Contains("Results"));
                subDirectories.Sort();
                string[] shapeFiles = System.IO.Directory.GetFiles(subDirectories[subDirectories.Count - 1]);

                List<string> pointShapeFiles = [];
                List<string> lineShapeFiles = [];
                List<string> polygonShapeFiles = [];

                //get all not currently displayed shape files
                try
                {
                    foreach (var file in shapeFiles)
                    {

                        if (file.Contains(".shp") && !file.Contains(".lock") && !file.Contains(".xml"))
                        {
                            if (file.Contains("Point"))
                            {
                                pointShapeFiles.Add(file);
                            }
                            else if (file.Contains("Line") || file.Contains("Vector") || file.Contains("Direction"))
                            {
                                lineShapeFiles.Add(file);
                            }
                            else if (file.Contains("Outline"))
                            {
                                polygonShapeFiles.Add(file);
                            }
                        }
                    }
                }
                catch (Exception e)
                {
                    MessageBox.Show(string.Format("Couldn't load Tree Tagger Shape Files from: {0}, {1}", shapeFilesPath, e.Message));
                    return;
                }

                if (pointShapeFiles.Count == 0 && lineShapeFiles.Count == 0 && polygonShapeFiles.Count == 0)
                {
                    MessageBox.Show("Couldn't find any shape files created by Tree Tagger");
                    return;
                }

                //add layers to project
                try
                {
                    ILayerContainerEdit groupContainer = group ? LayerFactory.Instance.CreateGroupLayer(map, 0, "TreeTagger") : map;

                    var pointSymbol = SymbolFactory.Instance.ConstructPointSymbol(ColorFactory.Instance.WhiteRGB, 5, SimpleMarkerStyle.Circle);

                    foreach (var sf in pointShapeFiles)
                    {
                        Uri pointFile = new(sf);
                        var pointLayer = LayerFactory.Instance.CreateLayer(pointFile, groupContainer) as FeatureLayer;

                        //Get the layer's current renderer
                        CIMSimpleRenderer renderer = pointLayer.GetRenderer() as CIMSimpleRenderer;

                        //Update the symbol of the current simple renderer
                        renderer.Symbol = pointSymbol.MakeSymbolReference();

                        //Update the feature layer renderer
                        pointLayer.SetRenderer(renderer);

                    }

                    //format line and polygon shape files to render correctly
                    var lineSymbol = SymbolFactory.Instance.ConstructLineSymbol(ColorFactory.Instance.RedRGB, 2, SimpleLineStyle.Solid);

                    foreach (var sf in lineShapeFiles)
                    {
                        Uri lineFile = new(sf);
                        var linesLayer = LayerFactory.Instance.CreateLayer(lineFile, groupContainer) as FeatureLayer;

                        //Get the layer's current renderer
                        CIMSimpleRenderer renderer = linesLayer.GetRenderer() as CIMSimpleRenderer;

                        //Update the symbol of the current simple renderer
                        renderer.Symbol = lineSymbol.MakeSymbolReference();

                        //Update the feature layer renderer
                        linesLayer.SetRenderer(renderer);
                    }

                    var outline = SymbolFactory.Instance.ConstructStroke(ColorFactory.Instance.CreateRGBColor(255, 0, 255, 30), 2.0, SimpleLineStyle.Solid);
                    var fillWithOutline = SymbolFactory.Instance.ConstructPolygonSymbol(ColorFactory.Instance.CreateRGBColor(255, 0, 255, 30), SimpleFillStyle.Solid, outline);

                    foreach (var sf in polygonShapeFiles)
                    {
                        Uri polyFile = new(sf);
                        var polygonLayer = LayerFactory.Instance.CreateLayer(polyFile, groupContainer) as FeatureLayer;

                        //Get the layer's current renderer
                        CIMSimpleRenderer rendererp = polygonLayer.GetRenderer() as CIMSimpleRenderer;

                        //Update the symbol of the current simple renderer
                        rendererp.Symbol = fillWithOutline.MakeSymbolReference();

                        //Update the feature layer renderer
                        polygonLayer.SetRenderer(rendererp);
                    }

                }
                catch (Exception e)
                {
                    MessageBox.Show(string.Format("Couldn't render shape files: {0}", e.Message));
                    return;
                }

            });
        }


    }
}

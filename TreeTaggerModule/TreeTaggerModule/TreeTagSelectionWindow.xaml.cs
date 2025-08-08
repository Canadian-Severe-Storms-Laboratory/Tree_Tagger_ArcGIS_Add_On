/* Main code for tree tagger add-on
 * 
 * Author: Daniel Butt NTP 2022
 * 
 * 
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Text.RegularExpressions;
using ArcGIS.Core.CIM;
using ArcGIS.Desktop.Mapping;
using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Core.Internal.CIM;
using System.IO;
using System.Text.Json;
using System.Net.Sockets;
using System.Text.Json.Serialization;
using System.Runtime.InteropServices;
using System.Threading;
using System.Diagnostics;

namespace TreeTaggerModule
{
    /// <summary>
    /// Interaction logic for TreeTagSelectionWindow.xaml
    /// </summary>
    public partial class TreeTagSelectionWindow : ArcGIS.Desktop.Framework.Controls.ProWindow
    {
        //current arcgis map
        private ArcGIS.Desktop.Mapping.Map map;
        //all raster layers on map
        private System.Collections.Generic.IEnumerable<RasterLayer> rLayers;
        //selected ploygon region
        private ArcGIS.Core.Geometry.Polygon polygon;

        private ListBox rasterSelectionBox;

        public TreeTagSelectionWindow(ArcGIS.Core.Geometry.Polygon poly)
        {
            InitializeComponent();

            polygon = poly;

            //get current map
            map = MapView.Active.Map;

            //get all raster layers on map
            rLayers = map.GetLayersAsFlattenedList().OfType<RasterLayer>();

            rasterSelectionBox = apiWindow.GetByUid("RasterSelectionBox") as ListBox;

            //add raster names to windows selection box
            foreach (var raster in rLayers)
            {
                rasterSelectionBox.Items.Add(raster.Name);
            }
            
            //disable selection box if a polygon region was already selected
            if(polygon != null)
            {
                rasterSelectionBox.IsEnabled = false;
            }

            apiWindow.ConstuctPacket = ConstructPacket;
            apiWindow.APICall = TreeTaggerAPI.AnalyzeEvent;
            apiWindow.AfterCall = AfterCall;

            //MessageBox.Show(Utils.AddinAssemblyLocation());
        }

        private async Task<Dictionary<string, Object>> ConstructPacket()
        {
            //getting paramters from textboxes
            double scale = 0.05;

            double angleThreshold = (apiWindow.GetByUid("angleThres") as NumberBox).GetNumber() / scale;
            double directAngleThreshold = (apiWindow.GetByUid("directAngleThres") as NumberBox).GetNumber() / scale;
            double directMergeThreshold = (apiWindow.GetByUid("directMergeThres") as NumberBox).GetNumber() / scale;
            double distThreshold = (apiWindow.GetByUid("distThres") as NumberBox).GetNumber() / scale;
            double minLengthThreshold = (apiWindow.GetByUid("minLength") as NumberBox).GetNumber() / scale;
            double maxLengthThreshold = (apiWindow.GetByUid("maxLength") as NumberBox).GetNumber() / scale;

            //whether a polygon region was selected
            bool isPolygon = polygon != null;
            bool _fast = (apiWindow.GetByUid("fast") as CheckBox).IsChecked ?? false;
            bool _joinMaps = (apiWindow.GetByUid("joinMaps") as CheckBox).IsChecked ?? false;
            bool _pointMaps = (apiWindow.GetByUid("pointMaps") as CheckBox).IsChecked ?? false;

            var polygonPoints = new List<double>();

            //if rasters were selected by name rather than using a polygon
            List<RasterLayer> selectedRasters = new List<RasterLayer>();
            if (!isPolygon)
            {
                foreach (string name in rasterSelectionBox.SelectedItems)
                {
                    foreach (var raster in rLayers)
                    {
                        if (raster.Name.Equals(name))
                        {
                            selectedRasters.Add(raster);
                        }
                    }
                }
            }

            var gridData = (apiWindow.GetByUid("VectorMapSelector") as VectorMapSelection).GetGridData();

            if (!isPolygon && selectedRasters.Count == 0)
            {
                MessageBox.Show("invaild raster selection");
                return null;
            }

            if (gridData.Item1.Count == 0 || gridData.Item2.Count == 0 || gridData.Item3.Count == 0)
            {
                MessageBox.Show("no direction grid sizes entered");
                return null;
            }


            Dictionary<string, Object> packet = new Dictionary<string, Object>();

            //This tell arcgis to run this code on the main application thread (not ui thread)
            //it is required whenever you need to access specific data from open arcgis project
            await QueuedTask.Run(() =>
            {
                //if a polygon was selected, get all rasters which intersect the polygon
                if (isPolygon)
                {
                    foreach (var pt in polygon.Points)
                    {
                        polygonPoints.Add(pt.X);
                        polygonPoints.Add(pt.Y);
                    }

                    var geoExtent = polygon.Extent;

                    foreach (var raster in rLayers)
                    {
                        var rExtent = raster.GetRaster().GetExtent();

                        if (rExtent.Intersects(geoExtent))
                        {
                            selectedRasters.Add(raster);
                        }
                    }
                }

                //gets the path to the currently open arcgis project
                var pathProject = System.IO.Path.GetDirectoryName(ArcGIS.Desktop.Core.Project.Current.URI);

                var rasterData = Utils.GetRasterData(selectedRasters);

                packet = new Dictionary<string, Object>
                {
                    { "assemblyPath", apiWindow.assemblyPath },
                    { "projectPath", pathProject },
                    { "polygonPoints", polygonPoints },
                    { "fast",  _fast},
                    { "joinMaps",  _joinMaps},
                    { "pointMaps", _pointMaps },
                    { "gridSizes", gridData.Item1},
                    { "minTrees", gridData.Item2},
                    { "averagingMethods", gridData.Item3},
                    { "angleThreshold", angleThreshold },
                    { "directMergeAngleThreshold", directAngleThreshold },
                    { "directMergeThreshold", directMergeThreshold },
                    { "distThreshold", distThreshold },
                    { "minLineLength", minLengthThreshold },
                    { "maxLineLength", maxLengthThreshold },
                    { "imagePaths", rasterData.Item1 },
                    { "imageCoords", rasterData.Item2 },
                    { "imageSizes", rasterData.Item3 },
                    { "imageScales", rasterData.Item4 }
                };

            });

            return packet;
        }

        private void AfterCall()
        {
            Utils.LoadShapeFiles(true);
        }

    }


}

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
using ArcGIS.Core.CIM;
using ArcGIS.Desktop.Mapping;
using ArcGIS.Desktop.Framework.Threading.Tasks;
using System.Xml.Linq;
using ArcGIS.Desktop.Internal.Mapping;

namespace TreeTaggerModule
{
    /// <summary>
    /// Interaction logic for PolygonReRun.xaml
    /// </summary>
    public partial class PolygonReRun : ArcGIS.Desktop.Framework.Controls.ProWindow
    {

        public PolygonReRun()
        {
            InitializeComponent();

            apiWindow.ConstuctPacket = ConstructPacket;
            apiWindow.APICall = TreeTaggerAPI.CreateOutlines;
            apiWindow.AfterCall = AfterCall;
        }

        private async Task<Dictionary<string, Object>> ConstructPacket()
        {
            var vectorData = await (apiWindow.GetByUid("vectorSelectionBox") as RasterSelectionBox).GetSelectedLayerData();

            if (vectorData.Item1 == null || vectorData.Item2.IsNullOrEmpty())
            {
                MessageBox.Show("Invalid vector shapefile selected");
                return null;
            }

            //(sizes, minTrees, smoothSteps, smoothGrowths, smoothShrinks);
            var outlineParams = (apiWindow.GetByUid("OutlineSelector") as OutlineSelection).GetData();

            //gets the path to the currently open arcgis project
            var pathProject = System.IO.Path.GetDirectoryName(ArcGIS.Desktop.Core.Project.Current.URI);

            Dictionary<string, Object> packet = new()
            {
                { "projectPath", pathProject },
                { "shapeFilePath", vectorData.Item1 },
                { "extent", vectorData.Item2 },
                { "gridSizes", outlineParams.Item1},
                { "minTrees", outlineParams.Item2},
                { "smoothSteps", outlineParams.Item3 },
                { "smoothGrowths", outlineParams.Item4 },
                { "smoothShrinks", outlineParams.Item5 }
            };

            return packet;
        }

        private void AfterCall()
        {
            Utils.LoadShapeFiles();
        }

    }
}


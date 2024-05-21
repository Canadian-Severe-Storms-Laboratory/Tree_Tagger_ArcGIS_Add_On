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
using ArcGIS.Desktop.Internal.Mapping;
using ArcGIS.Core.Internal.CIM;
using System.Net.Sockets;
using ArcGIS.Desktop.Framework.Controls;
using System.Text.Json;
using System.Diagnostics;

namespace TreeTaggerModule
{
    /// <summary>
    /// Interaction logic for DirectionReRunWindow.xaml
    /// </summary>
    /// 
    //list of direction grid sizes
    public partial class DirectionReRunWindow : ArcGIS.Desktop.Framework.Controls.ProWindow
    {

        public DirectionReRunWindow()
        {
            InitializeComponent();

            apiWindow.ConstuctPacket = ConstructPacket;
            apiWindow.APICall = TreeTaggerAPI.CreateVectorMaps;
            apiWindow.AfterCall = AfterCall;
        }


        private async Task<Dictionary<string, Object>> ConstructPacket()
        {
            bool _joinMaps = (apiWindow.GetByUid("joinMaps") as CheckBox).IsChecked ?? false;
            bool _pointMaps = (apiWindow.GetByUid("pointMaps") as CheckBox).IsChecked ?? false;

            var vectorData = await (apiWindow.GetByUid("vectorSelectBox") as RasterSelectionBox).GetSelectedLayerData();

            //gets the path to the currently open arcgis project
            var pathProject = System.IO.Path.GetDirectoryName(ArcGIS.Desktop.Core.Project.Current.URI);

            if (vectorData.Item1 == null || vectorData.Item2.IsNullOrEmpty())
            {
                MessageBox.Show("Invalid vector shapefile selected");
                return null;
            }

            var gridData = (apiWindow.GetByUid("VectorMapSelector") as VectorMapSelection).GetGridData();

            if (gridData.Item1.Count == 0 || gridData.Item2.Count == 0 || gridData.Item3.Count == 0)
            {
                MessageBox.Show("no direction grid sizes entered");
                return null;
            }

            Dictionary<string, Object> packet = new ()
            {
                { "projectPath", pathProject },
                { "shapeFilePath", vectorData.Item1 },
                { "extent", vectorData.Item2 },
                { "joinMaps",  _joinMaps},
                { "pointMaps", _pointMaps },
                { "gridSizes", gridData.Item1},
                { "minTrees", gridData.Item2},
                { "averagingMethods", gridData.Item3},
            };

            return packet;
        }

        private void AfterCall()
        {
            Utils.LoadShapeFiles();
        }

    }

    
}

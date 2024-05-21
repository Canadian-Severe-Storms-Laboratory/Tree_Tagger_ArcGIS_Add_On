using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Desktop.Mapping;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
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

namespace TreeTaggerModule
{
    /// <summary>
    /// Interaction logic for RasterSelectionBox.xaml
    /// </summary>
    public partial class RasterSelectionBox : UserControl
    {

        private System.Collections.Generic.IEnumerable<FeatureLayer> fLayers;

        public RasterSelectionBox()
        {
            InitializeComponent();

            //get current map
            var map = MapView.Active.Map;

            //get all raster layers on map
            fLayers = map.GetLayersAsFlattenedList().OfType<FeatureLayer>();

            //add raster names to windows selection box
            foreach (var layer in fLayers)
            {
                selectionBox.Items.Add(layer.Name);
            }
        }

        public FeatureLayer GetSelectedLayer()
        {
            if (selectionBox.SelectedItem == null) return null;

            string name = (string)selectionBox.SelectedItem;

            FeatureLayer selectedVector = null;

            foreach (var layer in fLayers)
            {
                if (layer.Name.Equals(name))
                {
                    selectedVector = layer;
                }
            }

            return selectedVector;
        }


        public async Task<(string, double[])> GetSelectedLayerData()
        {
            var layer = GetSelectedLayer();

            if (layer == null) return (null, null);

            string path = null;
            double[] extent = null;

            await QueuedTask.Run(() =>
            {
                path = layer.GetPath().ToString();

                if (path.Substring(0, 8).Equals(@"file:///"))
                {
                    path = path.Substring(8);
                }

                path = path.Replace("%20", " ");

                try
                {
                    if (layer == null || layer.GetFeatureClass() == null) return;

                    var geoExtent = layer.GetFeatureClass().GetExtent();
                    extent = [geoExtent.XMin, geoExtent.YMax, geoExtent.XMax, geoExtent.YMin];
                }
                catch (Exception _) { }

            });

            return (path, extent);
        }

    }
}

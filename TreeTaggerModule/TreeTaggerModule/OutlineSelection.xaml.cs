using System;
using System.Collections.Generic;
using System.Linq;
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
    /// Interaction logic for VectorMapSelection.xaml
    /// </summary>
    public partial class OutlineSelection : UserControl
    {
        //list of direction grid sizes
        private List<double[]> outlineParams = new List<double[]>();

        public OutlineSelection()
        {
            InitializeComponent();
        }

        public (List<int>, List<int>, List<int>, List<double>, List<double>) GetData()
        {
            var sizes = new List<int>();
            var minTrees = new List<int>();
            var smoothSteps = new List<int>();
            var smoothGrowths = new List<double>();
            var smoothShrinks = new List<double>();

            foreach (var gs in outlineParams)
            {
                sizes.Add((int)gs[0]);
                minTrees.Add((int)gs[1]);
                smoothSteps.Add((int)gs[4]);
                smoothGrowths.Add(gs[2]);
                smoothShrinks.Add(gs[3]);
            }

            return (sizes, minTrees, smoothSteps, smoothGrowths, smoothShrinks);
        }

        private void AddButtonClicked(object sender, RoutedEventArgs e)
        {
            double _gridSize = gridSize.GetIntNumber();
            double _gridMinTrees = gridMinTrees.GetIntNumber();
            double _growth = growth.GetNumber();
            double _shrink = shrink.GetNumber();
            double _steps = steps.GetIntNumber();


            if (_gridSize > 0 && _gridMinTrees > 0 && _growth > 0 && _shrink > 0 && _steps > 0)
            {
                outlineParams.Add([_gridSize, _gridMinTrees, _growth, _shrink, _steps]);
                gridSizesBox.Items.Add(string.Format("Size: {0}m, Min Trees: {1}, G: {2}, Sh: {3}, St: {4}", _gridSize, _gridMinTrees, _growth, _shrink, _steps));
            }
        }

        private void RemoveButtonClicked(object sender, RoutedEventArgs e)
        {
            if (gridSizesBox.SelectedIndex != -1)
            {
                outlineParams.RemoveAt(gridSizesBox.SelectedIndex);
                gridSizesBox.Items.RemoveAt(gridSizesBox.SelectedIndex);
            }
        }
    }
}

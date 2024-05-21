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
    public partial class VectorMapSelection : UserControl
    {
        //list of direction grid sizes
        private List<(int, int, int)> gridSizes = new List<(int, int, int)>();

        public VectorMapSelection()
        {
            InitializeComponent();
        }

        public (List<int>, List<int>, List<int>) GetGridData()
        {
            var sizes = new List<int>();
            var minTrees = new List<int>();
            var methods = new List<int>();

            foreach (var gs in gridSizes)
            {
                methods.Add(gs.Item1);
                sizes.Add(gs.Item2);
                minTrees.Add(gs.Item3);
            }

            return (sizes, minTrees, methods);
        }

        private void AddButtonClicked(object sender, RoutedEventArgs e)
        {
            int gridSizeI = gridSize.GetIntNumber();
            int gridMinTreesI = gridMinTrees.GetIntNumber();

            if (gridSizeI > 0 && gridMinTreesI > 0)
            {
                gridSizes.Add((directionInterpolationBox.SelectedIndex, gridSizeI, gridMinTreesI));
                gridSizesBox.Items.Add(string.Format("{0}, Size: {1}m, Min Trees: {2}", directionInterpolationBox.SelectedValue.ToString().Split(": ")[1], gridSizeI, gridMinTreesI));
            }
        }

        private void RemoveButtonClicked(object sender, RoutedEventArgs e)
        {
            if (gridSizesBox.SelectedIndex != -1)
            {
                gridSizes.RemoveAt(gridSizesBox.SelectedIndex);
                gridSizesBox.Items.RemoveAt(gridSizesBox.SelectedIndex);
            }
        }
    }
}

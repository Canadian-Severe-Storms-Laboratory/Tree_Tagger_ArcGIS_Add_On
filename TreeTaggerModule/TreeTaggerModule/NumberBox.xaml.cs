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
    /// Interaction logic for NumberBox.xaml
    /// </summary>
    public partial class NumberBox : UserControl
    {

        public string NumberString
        {
            get { return (string)GetValue(numberProperty); }
            set { SetValue(numberProperty, value); }
        }

        public static readonly DependencyProperty numberProperty = DependencyProperty.Register(
            "NumberString",
            typeof(string),
            typeof(NumberBox)
        );

        public NumberBox()
        {
            InitializeComponent();
        }

        public double GetNumber()
        {
            if (NumberString == null || NumberString.Length == 0)
            {
                return 0.0;
            }

            return double.Parse(NumberString, System.Globalization.CultureInfo.InvariantCulture);
        }

        public int GetIntNumber()
        {
            if (NumberString == null || NumberString.Length == 0)
            {
                return 0;
            }

            return (int)Math.Round(double.Parse(NumberString, System.Globalization.CultureInfo.InvariantCulture));
        }

        private void NumberValidationTextBox(object sender, TextCompositionEventArgs e)
        {
            string textBoxText = ((TextBox)sender).Text;

            char newChar = e.Text[0];

            bool foundDot = textBoxText.Contains('.');

            if (!char.IsDigit(newChar))
            {
                if (textBoxText.Length == 0 || newChar != '.' || foundDot)
                {
                    e.Handled = true;
                    return;
                }
            }

            e.Handled = false;

        }
    }
}

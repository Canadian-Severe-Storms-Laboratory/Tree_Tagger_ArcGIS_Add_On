using ArcGIS.Desktop.Framework.Threading.Tasks;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
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
    public partial class APIWIndow : UserControl
    {
        private TextBoxStreamWriter textBoxstreamWriter;
        private Dictionary<string, Object> packet;

        public string assemblyPath;
        public JsonSerializerOptions jsonSerializeOptions;

        public Func<Task<Dictionary<string, Object>>> ConstuctPacket;
        public Func<string, int> APICall;
        public Action AfterCall;

        public APIWIndow()
        {
            InitializeComponent();

            assemblyPath = Utils.AddinAssemblyLocation();

            jsonSerializeOptions = new() { WriteIndented = true };

            // Create an instance of the TextBoxStreamWriter and pass the TextBox
            textBoxstreamWriter = new TextBoxStreamWriter(ConsoleTextBox);
            textBoxstreamWriter.RedirectStandardOutput();
        }

        private async void DoneButtonClicked(object sender, RoutedEventArgs e)
        {
            if (ConstuctPacket == null || APICall == null || AfterCall == null)
            {
                MessageBox.Show("Error: Done Button not initizalized");
                return;
            }

            try
            {
                var parentWindow = Window.GetWindow(this);
                parentWindow.Closing += (object sender, CancelEventArgs e) => { this.Dispose(); };

                TreeTaggerAPI.cancelFlag = false;
                DoneButton.IsEnabled = false;
                textBoxstreamWriter.StartSpinning();

                packet = await ConstuctPacket();

                if (packet == null || packet.Count == 0) return;

                var jsonPacket = JsonSerializer.Serialize(packet, jsonSerializeOptions);

                int exitCode = 0;

                tabControl.SelectedIndex = 1;

                await QueuedTask.Run(() =>
                {
                    try
                    {
                        TreeTaggerAPI.PreLoadDLLs(assemblyPath);
                        exitCode = APICall(jsonPacket);
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(ex.Message);
                    }
                });

                textBoxstreamWriter.StopSpinning();

                if (exitCode == 0)
                {
                    AfterCall();
                }

                DoneButton.IsEnabled = true;
                
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        public void Dispose()
        {
            textBoxstreamWriter?.Close();
            TreeTaggerAPI.cancelFlag = true;
        }

        // LHS ?? RHS, means LHS unless LHS == null, then RHS instead
        public UIElement GetByUid(string uid, UIElement rootElement=null)
        {
            foreach (UIElement element in LogicalTreeHelper.GetChildren(rootElement ?? this).OfType<UIElement>())
            {
                if (element.Uid == uid) return element;

                UIElement resultChildren = GetByUid(uid, element);

                if (resultChildren != null) return resultChildren;
            }
            return null;
        }

        public UIElement ChildContent
        {
            get { return (UIElement)GetValue(ChildContentProperty); }
            set { SetValue(ChildContentProperty, value); }
        }

        public static readonly DependencyProperty ChildContentProperty = DependencyProperty.Register("ChildContent", typeof(UIElement), typeof(APIWIndow),
                                                                                            new PropertyMetadata(new PropertyChangedCallback(ChildContentChanged)));


        public static void ChildContentChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            APIWIndow uc = d as APIWIndow;
            uc.SettingsBorder.Child = e.NewValue as UIElement;
        }
    }
}

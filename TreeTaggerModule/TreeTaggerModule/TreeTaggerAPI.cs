

using ArcGIS.Desktop.Framework.Dialogs;
using Microsoft.Win32.SafeHandles;
using System;
using System.CodeDom;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace TreeTaggerModule
{
    internal static class TreeTaggerAPI
    {
        //"X:\\Treefall Model\\TreeTaggerModule\\TreeTaggerModule\\cppBin\\TreeTaggerOnnx.dll"
        //const string dllPath = "C:\\Users\\danie\\Documents\\Treefall Model\\TreeTaggerModule\\TreeTaggerModule\\cppBin\\";
        const string dllPath = "";

        [DllImport(dllPath + "TreeTaggerOnnx.dll", CallingConvention = CallingConvention.Cdecl)]
        private static unsafe extern int analyzeEvent(string packet, ConsoleCallback consoleCallback, IntPtr cancelFlag);

        [DllImport(dllPath + "TreeTaggerOnnx.dll", CallingConvention = CallingConvention.Cdecl)]
        private static unsafe extern int createVectorMaps(string packet, ConsoleCallback consoleCallback, IntPtr cancelFlag);

        [DllImport(dllPath + "TreeTaggerOnnx.dll", CallingConvention = CallingConvention.Cdecl)]
        private static unsafe extern int createOutlines(string packet, ConsoleCallback consoleCallback, IntPtr cancelFlag);

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern IntPtr LoadLibrary(string dllToLoad);

        public static bool cancelFlag = false;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        unsafe delegate void ConsoleCallback(string msg);

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        unsafe delegate void ProgressBarCallback(int inc);

        private static ConsoleCallback consoleCallback = (msg) => { Console.Write(msg); };

        private static IntPtr hModule;

        public static unsafe void PreLoadDLLs(string path)
        {
            string[] dlls = ["DirectML.dll", "opencv_world452.dll", "shp.dll", "onnxruntime.dll", "TreeTaggerOnnx.dll"];

            foreach (string dll in dlls)
            {
                hModule = LoadLibrary(path + "\\cppBin\\" + dll);

                if (hModule == IntPtr.Zero)
                {
                    //int errorCode = Marshal.GetLastWin32Error();
                    MessageBox.Show("Error, failed to load: " + dll);
                    return;
                }
            }

            //LoadLibrary(dllPath + "DirectML.dll");
        }

        private static unsafe int APICallWrapper(string packet, Func<string, ConsoleCallback, IntPtr, int> APICall)
        {
            cancelFlag = false;
            int result = -1;

            fixed (bool* cf = &cancelFlag)
            {
                result = APICall(packet, consoleCallback, (IntPtr)cf);
            }

            return result;
        }

        public static unsafe int AnalyzeEvent(string packet)
        {
            return APICallWrapper(packet, analyzeEvent);
        }

        public static unsafe int CreateVectorMaps(string packet)
        {
            return APICallWrapper(packet, createVectorMaps); 
        }

        public static unsafe int CreateOutlines(string packet)
        {
            return APICallWrapper(packet, createOutlines);
        }
    }
}

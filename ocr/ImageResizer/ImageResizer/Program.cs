using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace ImageResizer
{
    class Program
    {
        static void Main(string[] args)
        {
            const string imagesPath = @"C:\Users\Wojtek\Downloads\trained";
            const string csvFilePath = @"C:\Users\Wojtek\Downloads\file.csv";
            var dirInfo = new DirectoryInfo(imagesPath).GetDirectories("*", SearchOption.AllDirectories);

            var csv3 = new StringBuilder();
            csv3.Append("label");
            csv3.Append(',');
            for (int i = 0; i < 784; i++)
            {
                csv3.Append("pixel" + i);
                csv3.Append(',');
            }

            File.AppendAllText(csvFilePath, csv3 + "\n");

            foreach (var directoryInfo in dirInfo)
            {
                var files = directoryInfo.GetFiles("*", SearchOption.AllDirectories);
                Console.WriteLine("In folder : " + directoryInfo.FullName);
                for (int i = 0; i < 1920; i++)
                {
                    var csv = new StringBuilder();
                    var csv2 = new StringBuilder();
                    var fileName = files[i].DirectoryName.Replace(imagesPath + @"\", string.Empty);
                    csv.Append(fileName);
                    csv.Append(',');
                    var originalImage = new Image<Gray, byte>(files[i].FullName).Not();
                    var img = originalImage.Resize(28, 28, Inter.Linear);
                    for (var k = 0; k < img.Height; k++)
                    {
                        for (var j = 0; j < img.Width; j++)
                        {
                            csv.Append(img[k, j].Intensity);
                            csv.Append(',');
                        }
                    }

                    csv2.AppendLine(csv.ToString());
                    File.AppendAllText(csvFilePath, csv2.ToString());
                }
            }

            Console.WriteLine("DONE!");
            Console.ReadLine();

        }
    }

    public class CsvRow : List<string>
    {
        public string LineText { get; set; }
    }

    public class CsvFileWriter : StreamWriter
    {
        public CsvFileWriter(Stream stream)
            : base(stream)
        {
        }

        public CsvFileWriter(string filename)
            : base(filename)
        {
        }

        public void WriteRow(CsvRow row)
        {
            StringBuilder builder = new StringBuilder();
            bool firstColumn = true;
            foreach (string value in row)
            {
                // Add separator if this isn't the first value
                if (!firstColumn)
                    builder.Append(',');
                // Implement special handling for values that contain comma or quote
                // Enclose in quotes and double up any double quotes
                if (value.IndexOfAny(new char[] { '"', ',' }) != -1)
                    builder.AppendFormat("\"{0}\"", value.Replace("\"", "\"\""));
                else
                    builder.Append(value);
                firstColumn = false;
            }
            row.LineText = builder.ToString();
            WriteLine(row.LineText);
        }
    }
}

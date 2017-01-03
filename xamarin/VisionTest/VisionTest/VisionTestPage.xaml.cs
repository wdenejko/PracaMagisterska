using Xamarin.Forms;
using Microsoft.ProjectOxford.Vision;
using Microsoft.ProjectOxford.Vision.Contract;
using System.Collections.Generic;
using XLabs.Platform.Device;
using XLabs.Ioc;
using System.Text;
using System;
using System.Threading.Tasks;
using System.IO;
using Plugin.Media;
using Plugin.Media.Abstractions;

namespace VisionTest
{
	public partial class VisionTestPage : ContentPage
	{
		public VisionTestPage()
		{
			InitializeComponent();

			if (Device.OS == TargetPlatform.iOS)
				Padding = new Thickness(0, 25, 0, 0);
		}

		public async void TakePictureClicked(object sender, EventArgs e)
		{
			TakePictureButton.Text = "Working...";
			TakePictureButton.IsEnabled = false;
			ChoosePictureButton.IsEnabled = false;

			var VisionServiceClient = new VisionServiceClient("API_KEY_HERE");

			var photo = await TakePic();
			if (photo != null)
			{
				TakenImage.Source = ImageSource.FromFile(photo.Path);
				var ocrResult = await VisionServiceClient.RecognizeTextAsync(photo.GetStream(), "pl");
				RecognizedTextLabel.Text = this.LogOcrResults(ocrResult);
			}

			TakePictureButton.Text = "Take a picture";
			TakePictureButton.IsEnabled = true;
			ChoosePictureButton.IsEnabled = true;
		}

		public async void ChoosePictureClicked(object sender, EventArgs e)
		{
			ChoosePictureButton.Text = "Working...";
			TakePictureButton.IsEnabled = false;
			ChoosePictureButton.IsEnabled = false;

			var VisionServiceClient = new VisionServiceClient("API_KEY_HERE");

			var photo = await SelectPicture();
			if (photo != null)
			{
				TakenImage.Source = ImageSource.FromFile(photo.Path);
				var ocrResult = await VisionServiceClient.RecognizeTextAsync(photo.GetStream(), "pl");
				RecognizedTextLabel.Text = this.LogOcrResults(ocrResult);
			}

			ChoosePictureButton.Text = "...or choose from gallery";
			TakePictureButton.IsEnabled = true;
			ChoosePictureButton.IsEnabled = true;
		}

		private async Task<MediaFile> TakePic()
		{
			var mediaFile = await CrossMedia.Current.TakePhotoAsync(new StoreCameraMediaOptions
			{
				PhotoSize = PhotoSize.Custom,
				CustomPhotoSize = 50
			});

			return mediaFile;
		}

		private async Task<MediaFile> SelectPicture()
		{
			var mediaFile = await CrossMedia.Current.PickPhotoAsync(new PickMediaOptions
			{
				PhotoSize = PhotoSize.Custom,
				CustomPhotoSize = 50
			});

			return mediaFile;
		}

		protected string LogOcrResults(OcrResults results)
		{
			StringBuilder stringBuilder = new StringBuilder();

			if (results != null && results.Regions != null)
			{
				stringBuilder.AppendLine();
				foreach (var item in results.Regions)
				{
					foreach (var line in item.Lines)
					{
						foreach (var word in line.Words)
						{
							stringBuilder.Append(word.Text);
							stringBuilder.Append(" ");
						}

						stringBuilder.AppendLine();
					}

					stringBuilder.AppendLine();
				}
			}

			return stringBuilder.ToString();
		}

		private List<RecognizeLanguage> GetSupportedLanguages()
		{
			return new List<RecognizeLanguage>()
			{
				new RecognizeLanguage(){ ShortCode = "unk",     LongName = "AutoDetect"  },
				new RecognizeLanguage(){ ShortCode = "ar",      LongName = "Arabic"  },
				new RecognizeLanguage(){ ShortCode = "zh-Hans", LongName = "Chinese (Simplified)"  },
				new RecognizeLanguage(){ ShortCode = "zh-Hant", LongName = "Chinese (Traditional)"  },
				new RecognizeLanguage(){ ShortCode = "cs",      LongName = "Czech"  },
				new RecognizeLanguage(){ ShortCode = "da",      LongName = "Danish"  },
				new RecognizeLanguage(){ ShortCode = "nl",      LongName = "Dutch"  },
				new RecognizeLanguage(){ ShortCode = "en",      LongName = "English"  },
				new RecognizeLanguage(){ ShortCode = "fi",      LongName = "Finnish"  },
				new RecognizeLanguage(){ ShortCode = "fr",      LongName = "French"  },
				new RecognizeLanguage(){ ShortCode = "de",      LongName = "German"  },
				new RecognizeLanguage(){ ShortCode = "el",      LongName = "Greek"  },
				new RecognizeLanguage(){ ShortCode = "hu",      LongName = "Hungarian"  },
				new RecognizeLanguage(){ ShortCode = "it",      LongName = "Italian"  },
				new RecognizeLanguage(){ ShortCode = "ja",      LongName = "Japanese"  },
				new RecognizeLanguage(){ ShortCode = "ko",      LongName = "Korean"  },
				new RecognizeLanguage(){ ShortCode = "nb",      LongName = "Norwegian"  },
				new RecognizeLanguage(){ ShortCode = "pl",      LongName = "Polish"  },
				new RecognizeLanguage(){ ShortCode = "pt",      LongName = "Portuguese"  },
				new RecognizeLanguage(){ ShortCode = "ro",      LongName = "Romanian" },
				new RecognizeLanguage(){ ShortCode = "ru",      LongName = "Russian"  },
				new RecognizeLanguage(){ ShortCode = "sr-Cyrl", LongName = "Serbian (Cyrillic)" },
				new RecognizeLanguage(){ ShortCode = "sr-Latn", LongName = "Serbian (Latin)" },
				new RecognizeLanguage(){ ShortCode = "sk",      LongName = "Slovak" },
				new RecognizeLanguage(){ ShortCode = "es",      LongName = "Spanish"  },
				new RecognizeLanguage(){ ShortCode = "sv",      LongName = "Swedish"  },
				new RecognizeLanguage(){ ShortCode = "tr",      LongName = "Turkish"  }
			};
		}
	}

	public class RecognizeLanguage
	{
		public string ShortCode { get; set; }
		public string LongName { get; set; }
	}
}

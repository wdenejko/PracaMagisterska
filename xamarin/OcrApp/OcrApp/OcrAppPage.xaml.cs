using System;
using Xamarin.Forms;
using Tesseract;
using XLabs.Platform.Device;
using System.Threading.Tasks;
using XLabs.Platform.Services.Media;
using XLabs.Ioc;

namespace OcrApp
{
	public partial class OcrAppPage : ContentPage
	{
		private readonly ITesseractApi tesseractApi;
		private readonly IDevice device;
		
		public OcrAppPage()
		{
			InitializeComponent();
			
			if (Device.OS == TargetPlatform.iOS)
				Padding = new Thickness(0, 25, 0, 0);

			tesseractApi = Resolver.Resolve<ITesseractApi>();
			device = Resolver.Resolve<IDevice>();
		}

		public async void ChoosePictureClicked(object sender, EventArgs e) 
		{
			ChoosePictureButton.Text = "Working...";
			TakePictureButton.IsEnabled = false;
			ChoosePictureButton.IsEnabled = false;

			if (!tesseractApi.Initialized)
			{
				await tesseractApi.Init("eng");
			}

			var photo = await SelectPicture();
			if (photo != null)
			{
				var imageBytes = new byte[photo.Source.Length];
				photo.Source.Position = 0;
				photo.Source.Read(imageBytes, 0, (int)photo.Source.Length);
				photo.Source.Position = 0;

				TakenImage.Source = ImageSource.FromStream(() => photo.Source);
				var tessResult = await tesseractApi.SetImage(imageBytes);
				if (tessResult)
				{
					RecognizedTextLabel.Text = tesseractApi.Text;
				}
			}

			ChoosePictureButton.Text = "...or choose from gallery";
			TakePictureButton.IsEnabled = true;
			ChoosePictureButton.IsEnabled = true;
		}

		public async void TakePictureClicked(object sender, EventArgs e) 
		{
			TakePictureButton.Text = "Working...";
			TakePictureButton.IsEnabled = false;
			ChoosePictureButton.IsEnabled = false;
			
			if (!tesseractApi.Initialized)
			{
				await tesseractApi.Init("eng");
			}

			var photo = await TakePic();
			if (photo != null)
			{
				var imageBytes = new byte[photo.Source.Length];
				photo.Source.Position = 0;
				photo.Source.Read(imageBytes, 0, (int)photo.Source.Length);
				photo.Source.Position = 0;

				TakenImage.Source = ImageSource.FromStream(() => photo.Source);
				var tessResult = await tesseractApi.SetImage(imageBytes);
				if (tessResult)
				{
					RecognizedTextLabel.Text = tesseractApi.Text;
				}
			}

			TakePictureButton.Text = "Take a picture";
			TakePictureButton.IsEnabled = true;
			ChoosePictureButton.IsEnabled = true;
		}

		private async Task<MediaFile> TakePic()
		{
			var mediaStorageOptions = new CameraMediaStorageOptions
			{
				DefaultCamera = CameraDevice.Rear
			};
			var mediaFile = await device.MediaPicker.TakePhotoAsync(mediaStorageOptions);

			return mediaFile;
		}

		private async Task<MediaFile> SelectPicture() 
		{
			var mediaFile = await device.MediaPicker.SelectPhotoAsync(new CameraMediaStorageOptions
			{
				DefaultCamera = CameraDevice.Rear,
				MaxPixelDimension = 400
			});

			return mediaFile;
		}

	}
}

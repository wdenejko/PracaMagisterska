using Android.App;
using Android.Content.PM;
using Android.OS;
using TinyIoC;
using Tesseract;	
using Tesseract.Droid;
using XLabs.Ioc;
using XLabs.Ioc.TinyIOC;
using XLabs.Platform.Device;

namespace OcrApp.Droid
{
	[Activity(Label = "OcrApp.Droid", Icon = "@drawable/icon", MainLauncher = true, ConfigurationChanges = ConfigChanges.ScreenSize | ConfigChanges.Orientation)]
	public class MainActivity : global::Xamarin.Forms.Platform.Android.FormsApplicationActivity
	{
		protected override void OnCreate(Bundle bundle)
		{
			base.OnCreate(bundle);

			global::Xamarin.Forms.Forms.Init(this, bundle);

            var container = TinyIoCContainer.Current;
			container.Register<IDevice>(AndroidDevice.CurrentDevice);
			container.Register<ITesseractApi>((cont, parameters) =>
			{
				return new TesseractApi(ApplicationContext, Tesseract.Droid.AssetsDeployment.OncePerInitialization);
			});
			Resolver.SetResolver(new TinyResolver(container));

			LoadApplication(new App());
		}
	}
}

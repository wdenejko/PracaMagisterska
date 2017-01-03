
using System.IO;
using Android.App;
using Android.Content.PM;
using Android.Graphics;
using Android.OS;

using TinyIoC;
using XLabs.Ioc;
using XLabs.Ioc.TinyIOC;
using XLabs.Platform.Device;

using Plugin.Media;
using Plugin.Media.Abstractions;

namespace VisionTest.Droid
{
	[Activity(Label = "VisionTest.Droid", Icon = "@drawable/icon", Theme = "@style/MyTheme", MainLauncher = true, ConfigurationChanges = ConfigChanges.ScreenSize | ConfigChanges.Orientation)]
	public class MainActivity : global::Xamarin.Forms.Platform.Android.FormsAppCompatActivity
	{
		protected override void OnCreate(Bundle bundle)
		{

			base.OnCreate(bundle);

			global::Xamarin.Forms.Forms.Init(this, bundle);
			var container = TinyIoCContainer.Current;
			container.Register<IDevice>(AndroidDevice.CurrentDevice);
			Resolver.SetResolver(new TinyResolver(container));

			LoadApplication(new App());
		}
	}
}

using System;
using System.Collections.Generic;
using System.Linq;

using Foundation;
using UIKit;

using TinyIoC;
using XLabs.Platform.Device;
using XLabs.Ioc;
using XLabs.Ioc.TinyIOC;

namespace VisionTest.iOS
{
	[Register("AppDelegate")]
	public partial class AppDelegate : global::Xamarin.Forms.Platform.iOS.FormsApplicationDelegate
	{
		public override bool FinishedLaunching(UIApplication app, NSDictionary options)
		{
			global::Xamarin.Forms.Forms.Init();

			var container = TinyIoCContainer.Current;
			container.Register<IDevice>(AppleDevice.CurrentDevice);
			Resolver.SetResolver(new TinyResolver(container));
			LoadApplication(new App());

			return base.FinishedLaunching(app, options);
		}
	}
}

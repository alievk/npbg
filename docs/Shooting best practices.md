## Best practices of photo capture for photogrammetry

This page contains some guidelines and recommendations for taking photos for further photogrammetry via Agisoft Metashape, COLMAP, or different structure-from-motion (SfM) software.

This text is based both on observations of authors of Neural Point-Based Graphics and on the pieces of useful information available in the Internet. When preparing this text, we were additionally relying on the following sources:

* The Art of Photogrammetry: Introduction to Software and Hardware
by Brandon Blizard, https://www.tested.com/art/makers/460057-tested-dark-art-photogrammetry/, Feb. 11, 2014
* Interactive demo of exposure triangle by Tony Catalano, http://www.exposuretool.com/

### Guidelines

* **Choose a proper camera**

    The best results are obtained with the latest professional cameras, especially those equipped with full-frame sensor and with good video or continuous shooting capabilities (large resolution, high FPS). However, the latest smartphones can also be used for photogrammetry. They have smaller sensor => their depth-of-field can be made larger. On the downside, ISO cannot be set higher => background will be sharp, but picture will be noisy and lacking detail.

    Professional camera has larger sensor => you can set high ISO without noticing any noise, so detailization will be very high, however you can rarely set large depth-of-field => background might be blurry.

    While it's hard to say what is better in general &mdash; professional camera or a smartphone, the one can be said for sure: more modern smartphone is much better than an average smartphone, and more modern camera is much better than an average camera. When choosing a device, pay attention to the resolution it delivers, FPS and bitrate in video mode (or FPS in continuous shooting mode), focusing capabilities in video mode (or in continuous shooting mode), sensor size and other parameters.

    When using smartphone, make sure that you can set shutter speed and ISO in your camera app. We found that Android smartphones do not have such an option in the default camera app, so we used [OpenCamera](https://opencamera.org.uk/) app instead. All exposure parameters must not change during shooting. 

    When using professional camera (DSLR / mirrorless / etc.), it should be in Tv mode or, preferably, in M mode. Make sure that at least shutter speed and ISO are not set to AUTO.

* **Always keep your object in focus**

    When shooting video, always make sure to keep your object in focus. Many cameras do not refocus during video shooting &mdash; in this case you need to do it manually during the capture. Autofocus works differently on every setup, so manual focusing might be sometimes a better choice. Professional cameras usually feature various focus tracking modes, Face AF, Eye AF and other useful features. 

* **Shutter speed should be fast enough**

    For the case of photogrammetry, we found that shutter speed is the most important parameter of the [exposure triangle](https://fstoppers.com/education/exposure-triangle-understanding-how-aperture-shutter-speed-and-iso-work-together-72878). Insufficient shutter speed introduces motion blur which makes the photos completely useless for photogrammetry. Regardless of what camera or smartphone is used, we highly advise to find a shutter speed which does not introduce motion blur. Our observations show that 1/200 s &mdash; 1/400 s is enough when person is walking along a 360 degree trajectory for at least 30 seconds. 

* **Try the smallest ISO**

    On smartphones ISO of 200 can yield very high noise, while on DSLRs (especially with full frame matrices) ISO of 4000 can result in relatively low noise levels. It's always best to try it out with your camera and see which ISO yield critical levels. <!-- If not sure, look at DxOMark tests for your camera. -->

* **Use larger depth-of-field**

    When using a camera, setting smaller aperture (meaning, larger F-value) results in larger depth-of-field, which makes background sharper. On the contrary, lower F-values will only make an object of shooting sharp. When using Agisoft Metashape, we found that background can be beneficial for the first stage of cameras alignment, if it's sharp enough. Sharp background usually contains a lot of keypoints SfM can stick to when evaluating camera poses. While larger depth-of-field is arguably the least important, blurry background can make the cameras alignment worse. So, when possible, try to set the F-value larger. 

    This usually does not apply to smartphones, as due to its small sensor size the depth-of-field of a smartphone camera is often very large.

* **Try obtaining masks for background**

    When shutter speed is set fast enough and ISO is set low enough, sometimes it's impossible to set large F-value, since an image becomes too dark in this case (check out this [interactive demo of exposure triangle](http://www.exposuretool.com/)). One can find a way to mask the background somehow &mdash; either by using a constant color background (e.g. a "green screen"), or by segmenting an object on photos &mdash; either by hand or by software like Adobe Photoshop CC. In this case, there is no need to set large F-value.

    When the background is already sharp and you still want to improve quality by using segmentation masks, for Agisoft Metashape the best way to use masks would be to provide them by *Import masks...* dialogue, but not use them during alignment (*Apply masks to: None*). This way, they get automatically applied at the stage of point cloud generation, but do not interfere the alignment.

* **Good lighting**

    Lowering shutter speed, decreasing ISO and increasing F-value &mdash; all of this is making your image darker. If a careful balance of this parameters cannot be found, and masking the background does not help, you will likely require additional lighting in the room. Try shooting outdoors or adding some light sources.

* **Use enough photos**

    Too many images may overwhelm the software (especially if you don't have enough system RAM), but extra images will give you the luxury of picking the best shots after the fact. In our experiments, for 360 degrees flight around an object, 50 to 150 images was usually optimal. It's better when images more-or-less uniformly cover an object from all sides and there are no areas which are unconvered.

* **Make sure not to use any excessive compression, bad video codec**

    On all devices, check the camera settings to see if you are using the best quality available. On any camera, choose the largest JPEG quality or set the highest video bitrate. On DSLRs, no need to shoot in RAW as you won't use it. When extracting photos from a video, make sure to do it in [a lossless way (e.g. save in PNG)](https://stackoverflow.com/a/58672712) or in [the highest-quality JPEG](https://stackoverflow.com/a/10234065).

    <!-- TODO ffmpeg command to extract frames -->

### General algorithm

This is the algorithm which works in most situations:

1. Choose a camera, set it in **manual mode** (for Android smartphone &mdash; via using [OpenCamera](https://opencamera.org.uk/)), make sure you are using the highest quality, and all strange or extreme compression/enhancement algorithms are disabled. 
2. Make sure to set **fast shutter speed** to avoid motion blur (1/200 -- 1/400 s is usually enough, depending on how fast you walk during shooting).
3. Set **low enough ISO** which does not give you too noticeable noise under zoom-in. You need to take a few shots to find its critical ISO levels of your camera.
4. Set **F-value to the largest possible** (depending on distance to objects and your camera, it can be F6.3, F9 or larger value), such that the brightness is normal and the picture does not become too dark. Depth-of-field should at least fully cover the object of shooting.
5. Take a set of photos **while moving**. 

If you notice motion blur, try lowering shutter speed, if there is too much noise &mdash; lower ISO, too blurry background &mdash; increase F-value. Our experience shows that for photogrammetry shutter speed is more important than ISO, which is more important than F-value. 

In case you are unable to set all parameters such that image does not become too dark, consider trying **masking the background** (see above) or **improve your lighting** (buy some lamps or shoot outdoors, if possible).

If you think you followed all the instructions but the result is still bad, try to **play with the quality parameters of SfM software** you use and to follow the recommendations below.

### Recommendations

Understanding when SfM software works well is difficult &mdash; we've learnt it the hard way. Here are some recommendations which might help make a better 3D point cloud from your data.

* **Look at the photos**

    After taking the photos, make sure to zoom them in and ask yourself: "Do all the images convey enough detail to reconstruct a model from them? Do all the images convey the level of detail I would like to have in my model?". Feel free to delete any images which turned out bad for any reason &mdash; out of focus, blurry, etc. Most often, looking at images by yourself and deleting those which turned out noticeably wrong tremendously increases the SfM quality. 

* **No information is better than bad information**

    Give the software only high-confidence information. If you don't need the background and it's not razor sharp, mask it out. If you can't track a subject's hair, cover it up. If one image isn’t aligning correctly, get rid of it. You're smarter than the software at filtering this out before it gets to work, and you want to make its job as smooth as possible.   

* **There should be enough keypoints on all images**

    Images should contain more *keypoints* &mdash; corner points where contrast changes. This can be: colored textures with many changing patterns and colors, edgy objects, T-shirts with some text and patterns, complex background, etc. Shooting one or several objects having only monotonous color regions might significantly spoil the alignment -- SfM will find few corners or only corners with wrong correspondence between images. 

* **Objects should not move during the shooting and their appearance should not change**

    SfM expects objects to be looking consisently from all angles, as if all images were taken at the same point of time. Make sure to make images look like many cameras were taking them simultaneously. Do the best to avoid objects moving in front of camera, varying lighting and shadows, etc.

* **Don't choose too complex trajectory of shooting**

    While we successfully trajectories of various kind, we found that Agisoft works better when a circle-like trajectory is used and the radius is constant. This can be attributed to the fact that when you come farther from an object of shooting, the object starts occupying less space on an image, and fewer detail can be recovered from the image, thus worsening the alignment quality.


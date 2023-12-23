<h1 align="center">Jurassic Park through the Eyes of Neural Networks: Unveiling Diffusers for Image Generation in 10 Minutes</h1>

<p align="center">
  <img src="https://habrastorage.org/r/w1560/webt/1l/xy/ty/1lxytyjjukdkdfhltuwyzfb3bnu.png">
</p>

How often does it happen: you have a bit of free time, you want to relax and paint a picture... but either there's not enough time, or you simply don't feel like wielding a brush. However, you can delegate the task to a neural network – and you don't necessarily need to use Midjourney or DALL-E for that.

One option is to deploy your own assistant on a ready server using the Diffusers library and Hugging Face models. We tried it out and generated a whole "Jurassic Park" with various tyrannosaurs. We'll share the results and how to replicate our creations below.

### What You'll Need

Building your own "Jurassic Park" is not that difficult. You'll need a server with a GPU and a configured working environment with the Diffusers library.

### Tools

If you haven't worked with Diffusers before, now is the perfect time to get acquainted with it.

> *Diffusers is a library from Hugging Face that enables working with hundreds of pre-trained Stable Diffusion models for generating images, audio, and even volumetric molecular structures. It can be used for experimenting with existing models or training your own.*

Developers at Hugging Face claim that their creation is a straightforward modular project. Professional knowledge of neural network internals and "tensor magic" for working with Diffusers is not required.

The rest of the toolkit is classic. We will be using:
- JupyterLab: a development environment for Data Science and ML specialists,
- Python and specialized libraries: TensorFlow and PyTorch.

### Server

Generating even a single image can consume a significant amount of virtual resources and time. If you want to optimize the process and potentially integrate a Telegram bot interface with the neural network in the future, a local machine won't suffice. You'll need a server with a GPU—either dedicated or cloud-based.

A dedicated server offers more flexibility in configuration, including at the hardware level. Resources don't need to be shared with others—computational power works exclusively for you.

On the flip side, if something happens to the dedicated server, the recovery of all components might take a considerable amount of time. Cloud servers, on the other hand, are not tied to a specific host and can migrate if one fails. Additionally, you can always create an image of a cloud server, making it easier to restore the working environment. Therefore, for small projects, this is a more preferable option.

To avoid spending time on tool setup, you can deploy DAVM (Data Analytics Virtual Machine)—a cloud server with a pre-installed set of tools for data analysis and machine learning. This includes Jupyter Lab, Superset, and Prefect.

### Cloud Environment Setup

Setting Up the Cloud Server

Launching the DAVM server can be done in just a few steps:

1. Navigate to the Cloud Platform section within the control panel.

2. Select the ru-7a pool and create a cloud server with the Ubuntu 20.04 LTS Data Analytics 64-bit distribution and the required configuration.

   
> *It's crucial for the server to be accessible from the internet; otherwise, it won't be possible to connect from your computer. During the configuration setup, make sure to select a new public IP address.*

Allow the system a couple of minutes to boot. After that, connect to the server via SSH. You will find the credentials for accessing the DAVM environment in the console.

<p align="center">
  <a href="https://habrastorage.org/webt/9n/qi/8f/9nqi8fm_uijvuqodkyzg6l32q4m.gif">
    <img alt="Connection Details for DAVM." src="https://habrastorage.org/r/w1560/webt/75/7o/ad/757oadujs1-3-ltvbipxcispeoo.png">
  </a>
</p>

<p align="center">
  <i>Connection Details for DAVM.</i>
</p>




Now, by following the link in the message and logging into DAVM, you can launch JupyterLab, Keycloak, Prefect, or Superset.

### Connecting to DAVM:

<p align="center">
  <img src="https://habrastorage.org/r/w1560/webt/hp/yr/pa/hpyrpadfxnn5cl4xmzygi9i-kf0.png">
</p>

> *In DAVM, you can create users, manage them, and handle authentication in internal applications using Keycloak. Please note that after the initial change of the default password to your own, remote reset by technical support is not possible.*

**Dinosaur Generation**

Follow the link for a ready-made template that you can use for image generation, experimenting with models, and their parameters.

The logic of working with Diffusers models significantly simplifies project implementation. The template can be conditionally divided into several blocks: importing necessary libraries (including Diffusers), loading the model for dinosaur generation, configuring the pipeline, and displaying the image.

**Installing Dependencies**

To start, let's install the Diffusers library, as well as its related dependencies — transformers, scipy, accelerate.

```
! pip install diffusers transformers scipy
! pip install accelerate
```

Note: It's not necessary to pre-update the pip package manager; everything is already configured and packaged in the Docker container. Errors like "Could not install packages due to an OSError" are excluded.

**Loading the Model and Preparing the Pipeline**

Next, we'll import modules and load the model for dinosaur generation — for example, dreamlike-art/dreamlike-diffusion-1.0. This can be done by specifying the model_id, a variable-link to the model on Hugging Face, and using the StableDiffusionPipeline.from_pretrained() method, which will essentially prepare the pipeline:

```
model_id = "dreamlike-art/dreamlike-diffusion-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
```

**Small clarification:** You can use any generative model from the list. For instance, if you want an image in the style of Midjourney, load prompthero/openjourney. You can view the gallery of each model on Civitai and in the official library.

Great — you've chosen and loaded the model into the pipeline. Next, specify on which cores — CPU or GPU — you want to generate images. This can be done using the pipe.to() method. If you are using a server with a GPU, it should be pipe.to("cuda"), and if only CPU power is available, use pipe.to("cpu").

```
model_id = "dreamlike-art/dreamlike-diffusion-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
```

Next, you can customize the pipeline: input a prompt query, adjust parameters — for example, the number of images to generate in a single request, the number of iterations in inference, and more.

```
images = pipe(
    prompt = "photorealistic front view soft toy green dinosaur rex in a white T-shirt sat down behind a desk with computers and servers background is data centre",
    height = 512,
    width = 1024,
    num_inference_steps = 100,
    guidance_scale = 0.5,
    num_images_per_prompt = 1
).images

#img output
display(images[0])
```

Done — after running the program, it will display an image within Jupyter. In just a few minutes, we've set up a server with GPU and deployed a neural network for image generation — in our case, dinosaurs. Let's take a look at the beauties we've got.

**Results**

At the ML booth of the Selectel Tech Day, we invited participants to generate prompts using the neural network to create our mascot, Tyrex. Here are the coolest variations we came up with:

<p align="center">
  <img src="https://habrastorage.org/r/w1560/webt/cd/05/hd/cd05hdh8iirs2otxq6qpujfmmx0.png">
</p>
<p align="center">
  <img src="https://habrastorage.org/r/w1560/webt/rr/ea/h4/rreah4f3mffk1cyj7rtuv6vmtms.png">
</p>


I even got a Tyrex that closely resembles the plush version:

<p align="center">
  <img src="https://habrastorage.org/r/w1560/webt/di/md/hc/dimdhch-o1-tsp_qauddpwzwea4.png">
</p>


And here's a couple more dinosaurs playing chess:
<p align="center">
  <img src="https://habrastorage.org/r/w1560/webt/_o/jc/s7/_ojcs7_p47fb9apx-angkqeilte.png">
</p>

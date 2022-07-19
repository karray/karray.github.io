---
layout: post
title:  "Bringing Python to the Web"
date:   2022-07-12 12:01:27 +0200
tags: pyodide python web
---
Have you ever wanted to share your cool Python app with the world without deploying an entire Django server or developing a mobile app just for a small project?

Good news, you don’t have to! All you need is to add one JavaScript library to your HTML page and it will even work on mobile devices, allowing you to mix JS with Python so you can take advantage of both worlds.

<!--more-->

{% capture example1 %}
{% include posts/2022-07-12-bringing-python-to-the-web/simple_example.html %}
{% endcapture %}

{%include full_width.html content=example1%}

Witchcraft! This is made possible by [WebAssembly](https://webassembly.org/) (Wasm) and the [Pyodide project](https://github.com/iodide-project/pyodide). Full source can be found [here](https://github.com/karray/truepyxel/blob/master/pyodide.html).

So what can we actually do? Spoiler: With the power of Python and JS, we can do almost anything. But before getting into the details, let me first tell you a little story behind this writing.

I recently started a hobby [project](http://karay.me/truepyxel/) where I implemented image pixelation. I decided to write it in Python, as this language has a bunch of libraries for working with images. The problem was that I couldn't easily share the app without developing an Android app or finding a hosting and deploying a Django or Flask server.

I've heard about WebAssembly before and have wanted to try it out for a long time. Searching the Internet for "webassembly python", I immediately came across a link to an interesting article "[Pyodide: Bringing the scientific Python stack to the browser](https://hacks.mozilla.org/2019/04/pyodide-bringing-the-scientific-python-stack-to-the-browser/)". Unfortunately, the article is mainly about the [iodide project](https://github.com/iodide-project/iodide) that is no longer in development and the documentation of [Pyodide](https://github.com/iodide-project/pyodide) was sparse.

The idea to write this article came to me when I decided to contribute to the project by improving its [documentation](https://github.com/iodide-project/pyodide/blob/master/docs/using_pyodide_from_javascript.md#alternative-way-to-load-packages-and-run-python-code) after collecting the information about the API piece by piece and a number of experiments with code.

Here I would like to share my experience. I will also give more examples and discuss some issues.

# **What is Pyodide?**

Pyodide is a project from Mozilla that brings the Python runtime to the browser, along with the scientific stack including NumPy, Pandas, Matplotlib, SciPy, and [others](https://github.com/iodide-project/pyodide/tree/master/packages). All of this is made possible by Wasm.

> WebAssembly is a new type of code that can be run in modern web browsers and provides new features and major gains in performance. It is not primarily intended to be written by hand, rather it is designed to be an effective compilation target for source languages like C, C++, Rust, etc.

Wasm could potentially have a huge impact on the future of front-end development by extending the JS stack with numerous libraries and opening new possibilities for developers programming in languages other than JS. For example, there are already projects using it under the the hood, such as [PyScript](https://pyscript.net/).

So, it's time to get your hands dirty. Let's take a closer look at the minimal example

```html
<!DOCTYPE html>
<html>
<head>
<script src="https://pyodide-cdn2.iodide.io/v0.15.0/full/pyodide.js"></script>
<script>
   // init environment
  languagePluginLoader
    // then run Python code
    .then(() => console.log(pyodide.runPython(`import sys; sys.version`)));
</script>
</head>
<body>
</body>
</html>
```

First of all, we have to include the `pyodide.js` script by adding the CDN URL

```html
<!-- HTML -->
<script src="https://pyodide-cdn2.iodide.io/v0.15.0/full/pyodide.js"></script>
```

After this, we must wait until the Python environment is bootstrapped

```jsx
// JS
languagePluginLoader.then(...)
```

Finally, we can run Python code

```jsx
// JS
console.log(pyodide.runPython('import sys; sys.version'))
```

Note that if we want to load `pyodide.js` from a source other than the official CDN (e.g. own server), we have to set the base Plugin URL before including the `pyodide.js` as follows

```html
<!-- HTML -->
<script type="text/javascript">
	// default pyodide files URL (packages.json, pyodide.asm.data etc)
  window.languagePluginUrl = 'https://pyodide-cdn2.iodide.io/v0.15.0/full/';
</script>
<script src="https://pyodide-cdn2.iodide.io/v0.15.0/full/pyodide.js"></script>
```

This sets the path for downloading Python packages.

By default, the environment only includes standard Python modules such as `sys`, `csv`, etc. If we want to import a third-party package like `numpy` we have two options: we can either pre-load required packages manually and then import them in Python

```jsx
// JS
pyodide.loadPackage('numpy').then(() => {
	// numpy is now available
  pyodide.runPython('import numpy as np')
  console.log(pyodide.runPython('np.ones((3, 3)))'))
})
```

or we can use the `pyodide.runPythonAsync` function that will automatically download all packages that the code snippet imports

```java
// JS
python_code = `
import numpy as np
np.ones((3,3))
`
pyodide.runPythonAsync(python_code)
  .then(output => console.log(output))
```
{% include alert.html type='warning' title='Note' message='although the function is called `Async`, it still blocks the main thread. To run Python code asynchronously, we can use [WebWorkers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API).' %}

Okay, but how can we use all of this? In fact, we can replace JS and use Python as the main language for web development. Pyodide provides a bridge between JS and Python scopes.

# **Accessing JavaScript scope from Python**

The JS scope can be accessed from Python through the `js` module. This module gives us access to the global object `window` and allows us to directly manipulate the DOM and access global variables and functions from Python.

Why not try it yourself? Let's open the [demo page](http://karay.me/truepyxel/demo.html) in a new tab.

**Please be aware that execution of the code may take a while and the UI thread will be blocked until all packages have been downloaded.**

Just run this Python code and watch what happens.

```python
# Python
from jsimport documentdiv = document.createElement('div')
div.innerHTML = '<h1>This element was created from Python</h1>'
#insert into body as a first child
document.body.prepend(div)
```

We have just created an `h1` heading at the top of the page from Python. Isn't it cool?!

We first created a `div` element and then inserted it into the `body` using the JS `document` interface.

Since we have full control over the `window` object, we can also handle all events. Let's add a button at the bottom of the page that clears the output when clicked

```python
# Python
from js import document

def handle_clear_output(event):
  output_area = document.getElementById('output')
  output_area.value = ''

clear_button = document.createElement('button')
clear_button.innerHTML = 'Clear output'
clear_button.onclick = handle_clear_output
document.body.appendChild(clear_button)
```

Bear in mind, that we can only access the properties of the `window` object. That is, we can access only the variables directly attached to the window or defined globally with the `var` statement. Because `let` statement declares a block-scoped local variable just like the `const`, it does not create properties of the window object when declared globally.

Moreover, although the `js` module is an alias for the `window`, there is an [issue](https://github.com/iodide-project/pyodide/issues/768) with binding it to the `window` context. The workaround is to explicitly import the object as follows

```python
# Python
from js import window
```

JS has the arrow function expression introduced in ES6, which is very handy if we want to create a callback inline. An alternative in Python is the `lambda` expression. Let's take a look at one more example

```python
# Python
from js import window
window.fetch('http://karay.me/truepyxel/test.json').then(lambda resp: resp.json()).then(lambda data: data.msg).catch(lambda err: 'there were error: '+err.message)
```

I personally find this example very cool. Here we write the code in JS way and take advantage of chains of promises. The `resp.json` function converts the response body into an object that we can then access from Python. This also enables us to handle rejections. Just try to give any wrong URL to get the exception message.

The key difference is that it is not a real `Promise`. Therefore, the chaining will execute synchronously and the last value in the chain will be returned instead of a new `Promise`. Besides, as the project is still under development, there are some [issues](https://github.com/iodide-project/pyodide/issues/769). For example, we cannot use `Promise.finally` as this keyword is reserved in Python.

# **Accessing Python scope from JS**

We can also go in the opposite direction and get full access to the Python scope from JS through the `pyodide.globals` object. For example, if we import `numpy` into the Python scope, we can immediately use it from JS. This option is for those who prefer JS but want to take advantage of Python libraries.

Let's try it live. Go to the [demo](http://karay.me/truepyxel/demo.html) page and run the following Python code

```python
# Python
import numpyas npx = np.ones([3,3])
```

Now, while you are on the demo page, I will ask you to open the browser console and run this JS code

```jsx
// JS
pyodide.globals.x
// >>> [Float64Array(3), Float64Array(3), Float64Array(3)]
```

As we can see, the `x` variable was converted to JS typed array. We can also create the same array from JS:

```jsx
// JS
let x = pyodide.globals.np.ones(new Int32Array([3, 3]))
// x >>> [Float64Array(3), Float64Array(3), Float64Array(3)]
```

The `np.ones` function takes a size of the array as an argument, which must be a list or a tuple. If we pass in a standard JS array, we get an error because it won't be converted to a Python type. Therefore, we have to pass a [typed array](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Typed_arrays) (see Pyodide [Type conversion](https://github.com/iodide-project/pyodide/blob/master/docs/type_conversions.md) for more details).

Since we have full scope access, we can also re-assign new values or even JS functions to variables and create new ones from JS. Feel free to experiment with the code in the browser console.

```jsx
// JS

// re-assign a new value to an existing Python variable
pyodide.globals.x = 'x is now string'
// create a new js function that will be available from Python
// this will show a browser alert if the function is called from Python and msg is not null (None in Python)
pyodide.globals.alert = msg => msg && alert(msg)
// this new function will also be available in Python and will return the square of the window
pyodide.globals.window_square = function(){
  return innerHeight*innerWidth
}
```

All of these variables and functions will be available in the global Python scope

```python
# Python
alert('Hi from Python. Windows square: ' + str(window_square()))
```

# **Installing packages**

If we want to import a module that is not in the Pyodide repository, say `seaborn`, we will get the following error

```python
# Python
import seabornas sb
# => ModuleNotFoundError: No module named 'seaborn'
```

Pyodide currently supports a limited number of [packages](https://github.com/iodide-project/pyodide/tree/master/packages), but you can install the unsupported ones yourself using `micropip` module

```python
# Python
import micropip

micropip.install('seaborn').then(lambda msg: print('Done. You can now import the module'))
```

But this does not guarantee that the module will work correctly. Also, note that there must be a wheel file in [PyPi](https://pypi.org/) to install a module. More detailed information can be found [here](https://github.com/iodide-project/pyodide/blob/master/docs/pypi.md).

# **Advanced example**

Finally, let's look at the last example. Here we will create a plot using `matplotlib` and display it on the page. You can reproduce the result by running the following code on the [demo page](http://karay.me/truepyxel/demo.html).

First, we import all necessary modules. Since this will load a bunch of dependencies, the import will take a few minutes. The download progress can be seen in the browser console.

```python
# Python
from js import document
import numpy as np
import scipy.statsas stats
import matplotlib.pyplotas plt
import io, base64
```

The `numpy` and `scipy.stats` modules are used to create a Probability Density Function (PDF). The `io` and `base64` modules are used to encode the plot into a Base64 string, which we will later set as the source for an `<img>` tag.

Now let's create the HTML layout

```python
# Python

div_container = document.createElement('div')
div_container.innerHTML = """
  <br><br>
  mu:
  <input id='mu' value='1' type="number">
  <br><br>
  sigma:
  <input id='sigma' value='1' type="number">
  <br><br>
  <buttononclick='pyodide.globals.generate_plot_img()'>Plot</button>
  <br>
  <img id="fig" />
"""
document.body.appendChild(div_container)
```

The layout is pretty simple. The only thing I want to draw your attention to is that we have set `pyodide.globals.generate_plot_img()` as button's `onclick` handler. Since we create HTML as a string, we cannot access the Python scope. We can only assign a Python function as a handler directly if we created the button programmatically using the `document.createElement` function.

After that, we define the handler function itself

```python
# Python

def generate_plot_img():
	# get values from inputs
  mu = int(document.getElementById('mu').value)
  sigma = int(document.getElementById('sigma').value)
	# generate an interval
  x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
	# calculate PDF for each value in the x given mu and sigma and plot a line
  plt.plot(x, stats.norm.pdf(x, mu, sigma))
	# create buffer for an image
  buf = io.BytesIO()
	# copy the plot into the buffer
  plt.savefig(buf, format='png')
  buf.seek(0)
	# encode the image as Base64 string
  img_str = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('UTF-8')
	# show the image
  img_tag = document.getElementById('fig')
  img_tag.src = img_str
  buf.close()
```

This function will generate a plot and encode it as a Base64 string, which will then be set to the `img` tag.

You should get the following result:

![https://miro.medium.com/max/1400/1*byZ6FoML4TfhXT-Qn7RSjQ.png](https://miro.medium.com/max/1400/1*byZ6FoML4TfhXT-Qn7RSjQ.png)

Every time we click the button the `generate_plot_img` is called. The function gets values from the inputs, generates a plot, and sets it to the `img` tag. Since the `plt` object is not closed, we can add more charts to the same figure by changing the `mu` and `sigma` values

![https://miro.medium.com/max/1400/1*Kfp6Fw2IjrSKuyBkvXe8Zw.png](https://miro.medium.com/max/1400/1*Kfp6Fw2IjrSKuyBkvXe8Zw.png)

# **Conclusion**

Thanks to Pyodide, we can mix JS and Python and use the two languages interchangeably, allowing us to get the best of both worlds and speed up prototyping.

On the one hand, it enables us to extend JS with vast numbers of libraries. On the other hand, it gives us the power of HTML and CSS to create a modern GUI. The final application can then be shared as a single HTML document or uploaded to any free hosting service such as the GitHub pages.

There are of course some limitations. Apart from some of the issues discussed earlier, the main one is multithreading. This can be partially solved using WebWorkers.

As mentioned at the beginning, the Iodide project is no longer in development. The Pyodide is a subproject of Iodide and it is [still supported](https://github.com/iodide-project/pyodide/issues/766) by its community, so I encourage everyone to contribute to the project.

Wasm is a great technology that opens many possibilities. There are already a lot of interesting ports allowing to run games such as [Doom 3](http://www.continuation-labs.com/projects/d3wasm/#online-demonstration) and [Open Transport Tycoon Deluxe](https://milek7.pl/openttd-wasm/) inside modern Web Browsers. [MediaPipe](https://google.github.io/mediapipe/getting_started/javascript) allows us to process live media streams using ML on a webpage.

Furthermore, [WebAssembly System Interface (WASI)](https://github.com/bytecodealliance/wasmtime/blob/main/docs/WASI-intro.md) makes it possible to take full advantage of Wasm outside the browser:

> It's designed to be independent of browsers, so it doesn't depend on Web APIs or JS, and isn't limited by the need to be compatible with JS. And it has integrated capability-based security, so it extends WebAssembly's characteristic sandboxing to include I/O.
> 

For example, WASI enables us to import modules written in any language into [Node.js](https://nodejs.org/api/wasi.html) or into other languages (e.g. import Rust module into Python).

<!-- As Docker creator tweeted, WebAssembly has significant potential to become a Docker alternative -->

<!-- [https://twitter.com/solomonstre/status/1111004913222324225](https://twitter.com/solomonstre/status/1111004913222324225) -->
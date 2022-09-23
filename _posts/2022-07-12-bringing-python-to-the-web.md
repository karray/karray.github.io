---
layout: post
title:  "Bringing Python to the Web"
date:   2022-07-12 12:01:27 +0200
tags: pyodide python web
---
Have you ever wanted to share your cool Python app with the world without deploying an entire Django server or developing a mobile app just for a small project?

Good news, you don’t have to! All you need is to add one JavaScript library to your HTML page and it will even work on mobile devices, allowing you to mix JS with Python so you can take advantage of both worlds.

<!--more-->

Take a look at this REPL example:

{% capture example1 %}
{% include posts/2022-07-12-bringing-python-to-the-web/simple_example.html %}
{% endcapture %}

{%include full_width.html content=example1%}

{% include alert.html type='info' title='Note' message='This guide has been updated to Pyodide v0.21.3.' %}


Witchcraft! This is made possible by [WebAssembly](https://webassembly.org/) (Wasm) and the [Pyodide project](https://github.com/iodide-project/pyodide). You can also open the [pyodide_repl.html](/examples/pyodide_repl.html){:target='_blank'} ([source code](https://github.com/karray/karray.github.io/blob/master/examples/pyodide_repl.html)) example in a new tab.

So what can we actually do? Spoiler: With the power of Python and JS, we can do almost anything. But before getting into the details, let me first tell you a little story behind this writing.

I recently started a hobby [project](http://karay.me/truepyxel/) where I implemented image pixelation. I decided to write it in Python, as this language has a bunch of libraries for working with images. The problem was that I couldn't easily share the app without developing an Android app or finding a hosting and deploying a Django or Flask server.

I've heard about WebAssembly before and have wanted to try it out for a long time. Searching the Internet for "webassembly python", I immediately came across a link to an interesting article "[Pyodide: Bringing the scientific Python stack to the browser](https://hacks.mozilla.org/2019/04/pyodide-bringing-the-scientific-python-stack-to-the-browser/)". Unfortunately, the article is mainly about the [iodide project](https://github.com/iodide-project/iodide) that is no longer in development and the documentation of [Pyodide](https://github.com/iodide-project/pyodide) was sparse.

The idea to write this article came to me when I decided to contribute to the project by improving its [documentation](https://github.com/iodide-project/pyodide/blob/master/docs/using_pyodide_from_javascript.md#alternative-way-to-load-packages-and-run-python-code) after collecting the information about the API piece by piece and a number of experiments with code.

Here I would like to share my experience. I will also give more examples and discuss some issues.

# **What is Pyodide?**

> Pyodide was created in 2018 by [Michael Droettboom](https://github.com/mdboom) at Mozilla as part of the Iodide project. Iodide is an experimental web-based notebook environment for literate scientific computing and communication.

All of this is made possible by Wasm.

> WebAssembly is a new type of code that can be run in modern web browsers and provides new features and major gains in performance. It is not primarily intended to be written by hand, rather it is designed to be an effective compilation target for source languages like C, C++, Rust, etc.

Wasm could potentially have a huge impact on the future of front-end development by extending the JS stack with numerous libraries and opening new possibilities for developers programming in languages other than JS. For example, there are already projects using it under the hood, such as [PyScript](https://pyscript.net/) by Anaconda.

So, it's time to get your hands dirty. Let's take a closer look at the minimal example

```html
<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/pyodide/v0.21.3/full/pyodide.js"></script>
<script>
  (async () => { // create anonymous async function to enable await
    const pyodide = await loadPyodide();
    console.log(pyodide.runPython(`
import sys
sys.version
    `));
  })(); // call the async function immediately
</script>
</head>
<body>
</body>
</html>
```

First of all, we have to include the `pyodide.js` script by adding the CDN URL

```html
<!-- HTML -->
<script src="https://cdn.jsdelivr.net/pyodide/v0.21.3/full/pyodide.js"></script>
```

After this, we must load the main Pyodide wasm module using [loadPyodide](https://pyodide.org/en/stable/usage/api/js-api.html#globalThis.loadPyodide) and wait until the Python environment is bootstrapped

```js
const pyodide = await loadPyodide()
```

Finally, we can run Python code

```js
console.log(pyodide.runPython('import sys; sys.version'))
```

<!-- Note that if we want to load `pyodide.js` from a source other than the official CDN (e.g. own server), we have to set the base Plugin URL before including the `pyodide.js` as follows
This sets the path for downloading Python packages. -->

By default, the environment only includes standard Python modules such as `sys`, `csv`, etc. If we want to import a third-party package like `numpy` we have two options: we can either pre-load required packages manually and then import them in Python

```js
await pyodide.loadPackage('numpy');
// numpy is now available
pyodide.runPython('import numpy as np')
// create a numpy array
np_array = pyodide.runPython('np.ones((3, 3))')
// convert Python array to JS array
np_array = np_array.toJs()
console.log(np_array)
```

or we can use the [pyodide.loadPackagesFromImports](https://pyodide.org/en/stable/usage/api/js-api.html#pyodide.loadPackagesFromImports) function that will automatically download all packages that the code snippet imports

```js
const python_code = `
import numpy as np
np.ones((3,3))
`;
(async () => {
  await pyodide.loadPackagesFromImports(python_code)
  const result = pyodide.runPython(python_code)
  console.log(result.toJs())
})() // call the function immediately
```

{% include alert.html type='warning' title='Note' message='Since pyodide 0.18.0, [pyodide.runPythonAsync](https://pyodide.org/en/stable/usage/api/js-api.html#pyodide.runPythonAsync) does not automatically load packages, so `loadPackagesFromImports` should be called beforehand. It currently does not download packages from PyPI, but only downloads packages included in the Pyodide distribution (see [Packages list](https://pyodide.org/en/stable/usage/packages-in-pyodide.html#packages-in-pyodide)). More information about loading packages can be found [here](https://pyodide.org/en/stable/usage/loading-packages.html.)' %}

<!-- {% include alert.html type='warning' title='Note' message='although the function is called `Async`, it still blocks the main thread. To run Python code asynchronously, we can use [WebWorkers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API).' %} -->

Okay, but how can we use all of this? In fact, we can replace JS and use Python as the main language for web development. Pyodide provides a bridge between JS and Python scopes.

# **Accessing JavaScript scope from Python**

The JS scope can be accessed from Python through the `js` module. This module gives us access to the global object `window` and allows us to directly manipulate the DOM and access global variables and functions from Python. In other words, `js` is an alias for `window`, so we can either use `window` by importing it `from the js import window` or just use `js` directly.

Why not try it yourself? You can either try it out in the live demo above or open the [demo](https://karay.me/examples/pyodide_repl.html) in a new tab.

<!-- **Please be aware that execution of the code may take a while and the UI thread will be blocked until all packages have been downloaded.** -->

Just run this Python code and watch what happens.

```python
from js import document

div = document.createElement('div')
div.innerHTML = '<h1>This element was created from Python</h1>'
document.getElementById('simple-example').prepend(div)
```

We have just created an `h1` heading at the top of the example's container using Python. Isn't it cool?!

We first created a `div` element and then inserted it into the `<div id='simple-example'>` using the JS `document` interface.

Since we have full control over the `window` object, we can also handle all events from python. Let's add a button at the bottom of the example that clears the output when clicked

```python
from js import document

def handle_clear_output(event):
  output_area = document.getElementById('output')
  output_area.value = ''

clear_button = document.createElement('button')
clear_button.innerHTML = 'Clear output'
clear_button.onclick = handle_clear_output
document.getElementById('simple-example').appendChild(clear_button)
```

Note that we now use a Python function as an event handler.

{% include alert.html type='info' title='Note' message='We can only access the properties of the `window` object. That is, we can access only the variables directly attached to the window or defined globally with the `var` statement. Because `let` statement declares a block-scoped local variable just like the `const`, it does not create properties of the window object when declared globally.' %}

# **HTTP requests**

Python has a built-in module called `requests` that allows us to make HTTP requests. However, it is still [not supported](https://pyodide.org/en/stable/project/roadmap.html#write-http-client-in-terms-of-web-apis) by Pyodide. Luckily, we can use the [Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API) to make HTTP requests from Python.

Pyodide used to support JS `then/catch/finally` promise functions and we could use `fetch` as follows:

```python
from js import window
window.fetch('https://karay.me/assets/misc/test.json')
      .then(lambda resp: resp.json()).then(lambda data: data.msg)
      .catch(lambda err: 'there were error: '+err.message)
```

I personally find this example very cool. JS has the arrow function expression introduced in ES6, which is very handy if we want to create a callback inline. An alternative in Python is the `lambda` expression. Here we write the code in JS way and take advantage of chains of promises. The `resp.json()` function converts the response body into an object that we can then access from Python. This also enables us to handle rejections. 

However, since 0.17, it integrates the implementation of `await` for [JsProxy](https://pyodide.org/en/stable/usage/api/python-api/ffi.html#pyodide.ffi.JsProxy). So when JS returns a `Promise`, it converts it to `Future` in Python, which allows us to use `await`, but this object has no `then/catch/finally` attributes and hence it is no longer possible to build chains like in older versions. This should be [fixed](https://github.com/pyodide/pyodide/issues/2923) in the future, but for now, we can use the `await` keyword to wait for the response:

```python
import json
from js import window

resp = await window.fetch('https://karay.me/assets/misc/test.json')
data = await resp.json()
print(type(data))
# convert JsProxy to Python dict
data = data.to_py()
json.dumps(data, indent=2)
```

{% include alert.html type='info' title='Note' message='Since the code is executed using [runPythonAsync](https://pyodide.org/en/stable/usage/api/js-api.html#pyodide.runPythonAsync) we can use `await` outside of a function.' %}

As you probably noticed, we had to convert the `JsProxy` object to a Python `dict` using [JsProxy.to_py](https://pyodide.org/en/stable/usage/api/python-api/ffi.html#pyodide.ffi.JsProxy.to_py). This is required when we communicate between JS and Python. However, some standard types do not need to be converted since this is done implicitly. You can find more information about this [here](https://pyodide.org/en/stable/usage/type-conversions.html).

<!-- The key difference is that it is not a real `Promise`. Therefore, the chaining will execute synchronously and the last value in the chain will be returned instead of a new `Promise`. Besides, as the project is still under development, there are some [issues](https://github.com/iodide-project/pyodide/issues/769). For example, we cannot use `Promise.finally` as this keyword is reserved in Python. -->

# **Accessing Python scope from JS**

We can also go in the opposite direction and get full access to the Python scope from JS through the [pyodide.globals.get()](https://pyodide.org/en/stable/usage/api/js-api.html?highlight=globals.get#pyodide.globals) function. Additionally, similar to Python's `JsProxy.to_py`, we also need to convert the returned object to JS type using [PyProxy.toJs](https://pyodide.org/en/stable/usage/api/js-api.html#PyProxy.toJs) (we've already done this in previous examples). For example, if we import `numpy` into the Python scope, we can immediately use it from JS. This option is for those who prefer JS but want to take advantage of Python libraries.

Let's try it live

```python
import numpy as np
x = np.ones([3,3])
```

Now, I will ask you to open the browser console and run this JS code

```js
pyodide.globals.get('x').toJs()
// >>> [Float64Array(3), Float64Array(3), Float64Array(3)]
```

To access Python scope from JS, we use the [pyodide.globals.get()](https://pyodide.org/en/stable/usage/api/js-api.html#pyodide.globals) that takes the name of the variable or class as an argument. The returned object is a `PyProxy` that we convert to JS using `toJs()`.

As you can see, the `x` variable was converted to JS typed array. In earlier version (prior to v0.17.0), we could directly access Python scope:

```js
let x = pyodide.globals.np.ones(new Int32Array([3, 3]))
// x >>> [Float64Array(3), Float64Array(3), Float64Array(3)]
```

Now, we have to manually convert the `shape` parameter into Python type using [pyodide.toPy](https://pyodide.org/en/stable/usage/api/js-api.html#pyodide.toPy) and then convert the result back to JS:

```js
let x = pyodide.globals.get('np').ones(pyodide.toPy([3,3])).toJs()
// x >>> [Float64Array(3), Float64Array(3), Float64Array(3)]
```

This may [change](https://github.com/pyodide/pyodide/pull/2906) in the future and hopefully, most types will be implicitly converted.

Since we have full scope access, we can also re-assign new values or even JS functions to variables and create new ones from JS using `globals.set` function. Feel free to experiment with the code in the browser console.

```js
// re-assign a new value to an existing Python variable
pyodide.globals.set('x', 'x is now string')
// create a new js function that will be available from Python
// this will show a browser alert if the function is called from Python and msg is not null (None in Python)
pyodide.globals.set('alert', msg => msg && alert(msg))
// this new function will also be available in Python and will return the square of the window
pyodide.globals.set('window_square', function(){
  return innerHeight*innerWidth
})
```

All of these variables and functions will be available in the global Python scope:

```python
alert(f'Hi from Python. Windows square: {window_square()}')
```

# **Installing packages**

If we want to import a module that is not in the Pyodide repository, say `seaborn`, we will get the following error

```python
import seabornas sb
# => ModuleNotFoundError: No module named 'seaborn'
```

Pyodide currently supports a limited number of [packages](https://github.com/iodide-project/pyodide/tree/master/packages), but you can install the unsupported ones yourself using [micropip](https://pyodide.org/en/stable/usage/api/micropip-api.html#micropip.install) module

```python
import micropip

await micropip.install('seaborn')
```

But this does not guarantee that the module will work correctly. Also, note that there must be a wheel file in [PyPi](https://pypi.org/) to install a module. 

> If a package is not found in the Pyodide repository it will be loaded from PyPI. Micropip can only load pure Python packages or for packages with C extensions that are built for Pyodide.

The recent major release ([0.21-release](https://blog.pyodide.org/posts/0.21-release/)) introduces improvements to the systems for building and loading packages. It is now much easier to build and use binary wheels that are not included in the distribution. It also includes a large number of popular packages, such as `bitarray`, `opencv-python`, `shapely`, and `xgboost`.

Detailed information on how to install and build packages can be found [here](https://pyodide.org/en/stable/development/new-packages.html).

# **Advanced example**

Finally, let's look at the last example. Here we will create a plot using `matplotlib` and display it a the page. You can reproduce the result by running the following code on the [demo page](https://karay.me/examples/pyodide_repl.html).

First, we import all necessary modules. Since this will load a bunch of dependencies, the import will take a few minutes. The download progress can be seen in the browser console.

```python
from js import document
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import io, base64
```

The `numpy` and `scipy.stats` modules are used to create a Probability Density Function (PDF). The `io` and `base64` modules are used to encode the plot into a Base64 string, which we will later set as the source for an `<img>` tag.

Now let's create the HTML layout

```python
div_container = document.createElement('div')
div_container.innerHTML = """
  <br><br>
  mu:
  <input id='mu' value='1' type="number">
  <br><br>
  sigma:
  <input id='sigma' value='1' type="number">
  <br><br>
  <button onclick='pyodide.globals.get("generate_plot_img")()'>Plot</button>
  <br>
  <img id="fig" />
"""
document.body.appendChild(div_container)
```

The layout is pretty simple. The only thing I want to draw your attention to is that we have set `pyodide.globals.get("generate_plot_img")()` as button's `onclick` handler. Here, we get the `generate_plot_img` function from the Python scope and imminently call it.

After that, we define the handler function itself

```python
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

{% capture example2 %}
{% include posts/2022-07-12-bringing-python-to-the-web/advanced_example.html %}
{% endcapture %}

{%include full_width.html content=example2%}

<!-- ![https://miro.medium.com/max/1400/1*byZ6FoML4TfhXT-Qn7RSjQ.png](https://miro.medium.com/max/1400/1*byZ6FoML4TfhXT-Qn7RSjQ.png) -->

Every time we click the button the `generate_plot_img` is called. The function gets values from the inputs, generates a plot, and sets it to the `img` tag. Since the `plt` object is not closed, we can add more charts to the same figure by changing the `mu` and `sigma` values.

<!-- ![https://miro.medium.com/max/1400/1*Kfp6Fw2IjrSKuyBkvXe8Zw.png](https://miro.medium.com/max/1400/1*Kfp6Fw2IjrSKuyBkvXe8Zw.png) -->

# **Conclusion**

Thanks to Pyodide, we can mix JS and Python and use the two languages interchangeably, allowing us to get the best of both worlds and speed up prototyping.

On the one hand, it enables us to extend JS with vast numbers of libraries. On the other hand, it gives us the power of HTML and CSS to create a modern GUI. The final application can then be shared as a single HTML document or uploaded to any free hosting service such as the GitHub pages.

There are of course some limitations. Apart from some of the issues discussed earlier, the main one is multithreading. This can be partially solved using WebWorkers.

As mentioned at the beginning, the Iodide project is no longer in development. The Pyodide is a subproject of Iodide and it is [still supported](https://github.com/iodide-project/pyodide/issues/766) by its community, so I encourage everyone to contribute to the project.

Wasm is a great technology that opens many possibilities. There are already a lot of interesting ports allowing to run games such as [Doom 3](http://www.continuation-labs.com/projects/d3wasm/#online-demonstration) and [Open Transport Tycoon Deluxe](https://milek7.pl/openttd-wasm/) inside modern Web Browsers. [MediaPipe](https://google.github.io/mediapipe/getting_started/javascript) allows us to process live media streams using ML on a webpage.

Furthermore, [WebAssembly System Interface (WASI)](https://github.com/bytecodealliance/wasmtime/blob/main/docs/WASI-intro.md) makes it possible to take full advantage of Wasm outside the browser:

> It's designed to be independent of browsers, so it doesn't depend on Web APIs or JS, and isn't limited by the need to be compatible with JS. And it has integrated capability-based security, so it extends WebAssembly's characteristic sandboxing to include I/O.

For example, WASI enables us to import modules written in any language into [Node.js](https://nodejs.org/api/wasi.html) or into other languages (e.g. import Rust module into Python).

<!-- As Docker creator tweeted, WebAssembly has significant potential to become a Docker alternative -->

<!-- [https://twitter.com/solomonstre/status/1111004913222324225](https://twitter.com/solomonstre/status/1111004913222324225) -->
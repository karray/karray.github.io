---
layout: post
title:  "Bringing Python to the Web"
date:   2022-07-12 12:01:27 +0200
tags: pyodide python web
---
Have you ever wanted to share your cool Python app with the world without deploying an entire Django server or developing a mobile app just for a small project?

Good news, you don’t have to! All you need is to add a single JavaScript library to your HTML page and this will work even on mobile devices mixing JS with Python and take advantage of both worlds for your webpage.
<!--more-->

{% include posts/2022-07-12-bringing-python-to-the-web/simple_example.html %}

Easy, right? This is made possible by [WebAssembly][WebAssembly] (Wasm) and the [Pyodide][Pyodide] project.
So what can we actually do? Spoiler: With the power of Python and JS, we can do almost anything. But before getting into the details, let me first tell you a little story behind this writing.

[WebAssembly]: https://webassembly.org/
[Pyodide]: https://github.com/iodide-project/pyodide
+++
authors = ["Carlos"]
title = 'How to make the hugo jupyter page'
date = '2025-09-14T17:45:23-04:00'
description = "How to make a jupyter opage with hugo"
tags = [
    "hugo",
    "markdown",
    
]
categories = [
    "Site",
    
]
series = ["Site"]
+++


```plain
jupyter nbconvert --to markdown path/to/notebook.ipynb
This produces a notebook.md but in order to make it a page that renders in hugo we’ll need to do a bit more.

2. Add Frontmatter
In your notebook, create a markdown cell as the first cell and add your frontmatter:

---
date: 2024-02-02T04:14:54-08:00
draft: false
title: Example
---
3. Check Folder Structure
Put your notebook inside of the folder you want your page to be, and rename the notebook as index.ipynb.

This assumes the following folder structure:

path/to/project
└── content
    └── category-name
        └── page-name
            └── index.ipynb <--- your notebook titled 'index'
            --- 1 png
            --- 2 png
            --- 3 png

move the images from the jupyer folder created and add them to the folder which is page-name above

```


```plain
https://jlumbroso.github.io/hugo-geekdoc-github-example/features/code-blocks/
```
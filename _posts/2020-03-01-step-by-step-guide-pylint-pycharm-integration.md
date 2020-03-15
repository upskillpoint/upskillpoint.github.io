---
title: "Pylint for linting your code"
date: 2020-03-01
categories: [Python Static Testing]
tags: [Pylint,Static code analysis]
excerpt: "A step by step guide on how to use Pylint"
header:
  teaser: /assets/images/2020/03/01/Pylint.jpg
  overlay_image: /assets/images/2020/03/01/Pylint.jpg
  show_overlay_excerpt: False
---
If you are hired to develop a production level python application then you have to adhere to certain code standards , often developers write their codes which look easy and understandable to them but actually it's not. So to maintain code quality we can use different code quality management tools.

If you worked with PyCharm, you will note that the inspections plugin that performs static analysis on your code is very effective in identifying errors in PEP-8. Yet in some cases this fails and can be replaced with pylint. This tutorial will direct you through the setting up of pylint in PyCharm step by step in Windows.

### What is Static code analysis?

The main task of static code analysis tool is to evaluate compiled computer code or source code analysis, so that bugs can be quickly found without running the program.

* Provides insight into code without executing it
* Can automate code quality maintenance in the early stages
* Can automate early on discovering security problems
* Rapidly executes in contrast with dynamic analysis 

Note: You already using it (if you use any IDE that already has static analyzers, Pycharm uses pep8 for example).


### What Static code analysis does for you?

* Code styling analysis
* Duplicate code detection
* Complexity analysis
* Comment styling analysis
* Security linting
* Unused code detection
* Error detection
* UML diagram creation

### 1. Locate your pylint installation

If you don't have `pylint` installed then try the command abover after installing `pylint` via pip.

<pre>
$ pip install pylint
</pre>

If it's installed then below commands will work for you.
For windows: 
<pre>
$ where pylint
.\Python\Python36\Scripts\pylint.exe
</pre> 

### 2. Go to External tools in PyCharm

You can find the *External Tools* options from the 

1. File -> Settings
2. Typing *External Tools* in the search bar
![Step image 2]({{ site.url }}/assets/images/2020/03/01/Step-2.jpg)

You can read more about *External Tools* [here](https://www.jetbrains.com/help/pycharm/2017.1/external-tools.html).

### 3. Add Pylint as an External Tool

Tap on the' +' button in the* External Tools* window and customize using the details below.

![Step image 3]({{ site.url }}/assets/images/2020/03/01/Step-3.png)

1. Program: *Use the path found in [Step 1](#1-locate-your-pylint-installation).*
2. Arguments: "--msg-template='{abspath}:{line:5d}:{column}: {msg_id}: {msg} ({symbol})'" --output-format=colorized "$FilePath$"
3. Working directory: $ProjectFileDir$
4. Output filters: $FILE_PATH$:\s*$LINE$\:\s*$COLUMN$:

### 4. Run Pylint on entire project or a single file

To run on a single file:

Run `pylint` from *External Tools*  via *Tools -> External Tools -> pylint* dropdown.

To run on the entire project:

1. Right click on the root directory.
2. Run `pylint` from *External Tools*  via External Tools -> pylint* dropdown.

### 5. Check your Pylint score on PyCharm console

After your run from [Step 4](#4-run-pylint-on-entire-project-or-a-single-file), you can view your `pylint` score in your PyCharm console.

Pylint prefixes each of the problem areas with an R, C, W, E, or F, meaning:

* [R]efactor for “good practice” metric violation
* [C]onvention for coding standard violation
* [W]arning for stylistic problems, or minor programming issues
* [E]rror for important programming issues (i.e. most probably a bug)
* [F]atal for errors which prevented further processing

Regarding the coding style, Pylint follows the PEP8 style guide.

### 6. Configure Pylint to as per your requirement

You can configure Pylint with a ~/.pylintrc file-this allows you to ignore warnings you don't care about, amongst other things.

1. At first generate the global configuration file in your project directory by below command:

<pre>
$ pylint --generate-rcfile > .pylintrc
</pre> 

2. Once it is generated you may put it in:

* /etc/pylintrc for default global configuration
* ~/.pylintrc for default user configuration
* <your project>/pylintrc for default project configuration (used when you'll run pylint <your project>)
* wherever you want, then use pylint --rcfile= Given path

Do remember when generating the rc file, you can add option to the command line before the —generate-rcfile to be included in the generated file.

I don't recommend a system-wide or user-wide rc file against this. Using it per project is almost always fine.

Example of a configuration change:

* To disable other warnings, use Message Control to turn them off individually:

<pre>
[MESSAGES CONTROL]
# C0111: Missing docstring
# R0904: Too many public methods
disable=C0111,R0904
</pre> 

Keep this config file in your project root folder and run Pylint.
*  Now Pylint would not check for these two problems.

Congratulations!! Now you are ready to write some powerful applications!!

Reference:

[Pylint](https://www.pylint.org/)
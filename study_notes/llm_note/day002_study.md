# Day 2 Study Plan: Python Collections, Classes, and File I/O

## Goal
Use common Python data structures and organization patterns while keeping a small PyTorch tensor program readable.

## Today's Focus
- Python `list` and `dict` operations
- Creating and using a simple class
- Reading from and writing to a text file
- Organizing yesterday's tensor code into reusable functions

## Study Tasks

### 1. Lists and Dictionaries
Learn or review:
- creating a `list`
- indexing and appending values
- iterating with a `for` loop
- storing related values in a `dict`
- reading values with dictionary keys

### 2. Classes
Learn or review:
- defining a class with `class`
- initializing attributes with `__init__`
- defining instance methods
- creating an object from a class

### 3. File I/O
Learn or review:
- opening files safely with `with`
- writing text with `write_text()`
- reading text with `read_text()`
- using `pathlib.Path` for file paths

### 4. Hands-on Exercise
Create a Python file and complete these tasks:
1. Create a list of study topics and print each topic.
2. Create a dictionary describing a study session.
3. Define a `StudySession` class that prints a summary.
4. Write the summary to a temporary text file and read it back.
5. Create two PyTorch tensors and print their addition and multiplication.
6. Put tensor reporting into a function.

## Example Code

```python
from pathlib import Path


class StudySession:
    def __init__(self, day, topics):
        self.day = day
        self.topics = topics

    def summary(self):
        return f"Day {self.day}: {', '.join(self.topics)}"


session = StudySession(2, ["lists", "classes", "file I/O"])
path = Path("session_summary.txt")
path.write_text(session.summary(), encoding="utf-8")
print(path.read_text(encoding="utf-8"))
```

## What You Should Understand After Today
- Lists store ordered collections, while dictionaries map keys to values.
- A class combines related data and behavior into reusable objects.
- `with` and `pathlib.Path` make file operations clear and safe.
- Functions and classes help keep PyTorch programs organized as they grow.

## Suggested Schedule
- 20 min: Practice lists and dictionaries
- 20 min: Learn classes and write one small class
- 20 min: Read and write a text file
- 30 min: Refactor and run the PyTorch exercise
- 10 min: Write study notes

## Mini Challenges
- Add a new topic to the list with `append()`.
- Add a `completed` field to the session dictionary.
- Add a method that returns the number of study topics.
- Save one tensor result to a file as text.
- Change the tensor shapes and observe the output.

## Notes Template

```markdown
## Day 2 Notes

- When would I use a list instead of a dictionary?
- What information belongs in a class?
- How does `with` help when working with files?
- How did functions make the tensor code easier to read?
- One thing I want to review tomorrow:
```

## Completion Checklist
- [ ] I created and iterated over a list.
- [ ] I created and read values from a dictionary.
- [ ] I wrote and used a class.
- [ ] I wrote and read a text file.
- [ ] I used functions to organize tensor code.
- [ ] I wrote short notes about today's study.

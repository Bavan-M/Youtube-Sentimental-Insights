from setuptools import find_packages,setup
from pathlib import Path

with open('README.md',"r",encoding="utf-8") as f:
    long_description=f.read()
__version__="0.0.0"

REPO_NAME="Youtube-Sentimental-Insights"
AUTHOR_USER_NAME="Bavan-M"
SRC_REPO="youtubeInsights"
AUTHOR_EMAIL="bavanreddy1999@gmail.com"
README=(Path(__file__).parent/"README.md").read_text(encoding="utf-8")

setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for Sentimental insights",
    long_description=long_description,
    long_description_content_type="text.markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker":f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    packages=find_packages()
)
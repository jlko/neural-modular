git add -u
git commit -m "${1}"
git push
git checkout public
git merge main
git add -u
git commit -m "$1"
git push
git checkout main
git push git@github.com:jlko/neural-modular.git public:main

local_path = $(shell pwd)
project_name := $(shell basename $(local_path))
local_scratch = /scratch-ssd/jansen/$(project_name)/

remote_oat = jansen@oatcloud:/users/jansen/
remote_scratch = jansen@clpc57:/scratch-ssd/jansen/
remote_arc = arc_htc:~/
remote_arc_data = arc_htc:/data/coml-ecr/newc6130

.PHONY: rsync_base
rsync_base:
	@(echo "from $(from_path) to $(to_path)")
	@(rsync -rahuv -zz --delete --progress  \
	--exclude data \
	--exclude outputs \
	--exclude legacy_outputs \
	--exclude "__pycache__" \
	--exclude *.cpython \
	--exclude "*.out" \
	--exclude ".swp" \
	--exclude clean-notebooks \
	--exclude ".git" \
	$(from_path) $(to_path))

.PHONY: sync-to-oat
sync-to-oat: from_path = $(local_path)
sync-to-oat: to_path = $(remote_oat)
sync-to-oat: rsync_base

.PHONY: s2o
s2o: sync-to-oat

.PHONY: sync-to-dell
sync-to-dell: from_path = $(local_path)
sync-to-dell: to_path = $(remote_dell)
sync-to-dell: rsync_base

.PHONY: sync-to-arc
sync-to-arc: from_path = $(local_path)
sync-to-arc: to_path = $(remote_arc)
sync-to-arc: rsync_base

.PHONY: sync-from-oat
sync-from-oat: from_path = $(remote_oat)/$(project_name)/
sync-from-oat: to_path = $(local_path)
sync-from-oat: rsync_base

.PHONY: sfo
sfo: sync-from-oat


.PHONY: rsync-outputs-base
rsync-outputs-base:
	@(rsync -rahuv -zz --progress \
	--exclude loss_data.pkl \
	--exclude loss_data.pkl_last \
	--exclude pmfs.pkl \
	--exclude *.pth \
	--exclude wandb/ \
	$(from_path)/outputs $(to_path)/)


.PHONY: sync-outputs-dell
sync-outputs-dell: from_path = $(remote_dell)/$(project_name)/
sync-outputs-dell: to_path = $(local_scratch)
sync-outputs-dell: rsync-outputs-base


.PHONY: sync-outputs-arc
sync-outputs-arc: from_path = $(remote_arc_data)/$(project_name)/
sync-outputs-arc: to_path = $(local_scratch)
sync-outputs-arc: rsync-outputs-base

.PHONY: sync-outputs-arc-home
sync-outputs-arc-home: from_path = $(remote_arc)/$(project_name)/
sync-outputs-arc-home: to_path = $(local_scratch)
sync-outputs-arc-home: rsync-outputs-base


.PHONY: sync-outputs-local
sync-outputs-local: from_path = $(local_path)/
sync-outputs-local: to_path = $(local_scratch)
sync-outputs-local: rsync-outputs-base


.PHONY: sync-outputs-all
sync-outputs-all:
	@(make sync-outputs-local)
	@(make sync-outputs-arc && make sync-outputs-arc-home)


.PHONY: sync-local-exhaustive
sync-local-exhaustive:
	@(rsync -rahuv -zz --progress \
	--exclude "*.out" \
	$(local_path)/ $(local_scratch)/)


.PHONY: sync-outputs-arc-old
sync-outputs-arc-old: from_path = $(remote_arc)/$(project_name)/
sync-outputs-arc-old: to_path = $(local_path)
sync-outputs-arc-old: rsync-outputs-base


.PHONY: nb-start
nb-start:
	@(jupyter notebook --no-browser --port=8888 --ip 127.0.0.1)


.PHONY: port-fwd
port-fwd:
	@(ssh -N -f -L localhost:8888:localhost:8888 jansen@clpc466)


.PHONY: ssh-dell
ssh-dell:
	@(ssh $(remote_dell))


.PHONY: sync-plots
sync-plots:
	@(rsync -rahu --delete --progress  \
	$(remote_oat)$(project_name)/notebooks/plots $(local_path)/notebooks/)


.PHONY: nb-commit
nb-commit:
	@(rm -rf clean-notebooks)
	@(rsync -ra --include "*.ipynb" --exclude ".ipynb_checkpoints" notebooks/ clean-notebooks/)
	@(jupyter nbconvert --clear-output	./clean-notebooks/*ipynb)
	@(jupyter nbconvert --clear-output	./clean-notebooks/*/*ipynb)
	@(jupyter nbconvert --clear-output	./clean-notebooks/*/*/*ipynb)
	@(git add -f clean-notebooks/*)
	@(git status)
	@(git commit -m "Updating notebooks.")
	@(git push)


.PHONY: sync-nb-back
sync-nb-back:
	@(rsync -rah --progress  \
	$(remote_path)$(project_name)/notebooks $(local_path)/ )



.PHONY: sync-scratch-to-external
sync-scratch-to-external:
	@(rsync -rah -zz --progress  \
	$(remote_oat)$(project_name)/outputs /Volumes/Toshiba/Oat/active-quadrature/ )

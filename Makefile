.PHONY: lint test gcp-create-infra gcp-deploy gcp-bootstrap gcp-install-cron gcp-health gcp-run-now

lint:
	ruff check tradingagents/system tests

test:
	pytest -q

gcp-create-infra:
	./scripts/gcp/create_infra.sh

gcp-deploy:
	./scripts/gcp/deploy_or_update.sh

gcp-bootstrap:
	./scripts/gcp/bootstrap_vm.sh

gcp-install-cron:
	./scripts/gcp/install_cron.sh

gcp-health:
	./scripts/gcp/health_check_vm.sh

gcp-run-now:
	./scripts/gcp/run_remote_daily.sh

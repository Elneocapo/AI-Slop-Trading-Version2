from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    trading_mode: str = "paper"
    trading_enabled: bool = False
    alpaca_paper: bool = True
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    ticker: str = "NVDA"
    initial_cash: float = 10000.0
    max_trade_amount: float = 250.0
    max_portfolio_exposure: float = 0.25
    max_daily_loss: float = 0.02
    max_open_positions: int = 3
    max_spread_percent: float = 10.0
    min_option_volume: int = 10
    min_open_interest: int = 25
    min_dte: int = 14
    max_dte: int = 60
    model_threshold: float = 0.60
    slippage_percent: float = 0.50
    commission_per_trade: float = 1.0
    database_url: str = "sqlite:///brokeria.db"
    log_level: str = "INFO"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def validate_safety(self) -> None:
        if self.trading_mode.lower() != "paper":
            raise RuntimeError("BrokerIA first release only permits TRADING_MODE=paper")
        if not self.alpaca_paper:
            raise RuntimeError("ALPACA_PAPER must remain true in the first release")


settings = Settings()
settings.validate_safety()

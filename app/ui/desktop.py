"""Tkinter desktop interface for BrokerIA.

The UI is deliberately thin: it calls the existing data, analysis and strategy
modules instead of duplicating trading logic. Network/model work runs in a
background thread so the window remains responsive.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

from app.analysis import add_indicators
from app.config import settings
from app.data import get_historical_prices, get_stock_quote
from app.strategy import generate_signal


class BrokerIAApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BrokerIA — Quantitative Research")
        self.geometry("1180x760")
        self.minsize(980, 650)
        self.configure(bg="#101318")
        self._build_style()
        self._build_ui()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#101318")
        style.configure("Card.TFrame", background="#181d24")
        style.configure("TLabel", background="#101318", foreground="#e8edf2", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"), foreground="#ffffff")
        style.configure("Muted.TLabel", foreground="#8e9aa8", font=("Segoe UI", 9))
        style.configure("Value.TLabel", background="#181d24", foreground="#ffffff", font=("Segoe UI", 18, "bold"))
        style.configure("Metric.TLabel", background="#181d24", foreground="#9eabb8", font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(16, 9))
        style.configure("TEntry", fieldbackground="#202630", foreground="#ffffff", insertcolor="#ffffff")
        style.configure("TCombobox", fieldbackground="#202630", foreground="#ffffff")

    def _build_ui(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x", padx=24, pady=(20, 10))
        ttk.Label(header, text="BrokerIA", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Quantitative research • PAPER TRADING ONLY", style="Muted.TLabel").pack(side="left", padx=16, pady=(8, 0))

        controls = ttk.Frame(self, style="Card.TFrame", padding=16)
        controls.pack(fill="x", padx=24, pady=8)

        ttk.Label(controls, text="Ticker").grid(row=0, column=0, sticky="w")
        self.ticker_var = tk.StringVar(value=settings.ticker)
        ttk.Entry(controls, textvariable=self.ticker_var, width=14).grid(row=1, column=0, padx=(0, 14), pady=(5, 0), sticky="w")

        ttk.Label(controls, text="Historical period").grid(row=0, column=1, sticky="w")
        self.period_var = tk.StringVar(value="2y")
        ttk.Combobox(controls, textvariable=self.period_var, values=("6mo", "1y", "2y", "5y"), state="readonly", width=10).grid(row=1, column=1, padx=(0, 14), pady=(5, 0), sticky="w")

        self.run_button = ttk.Button(controls, text="Run analysis", style="Accent.TButton", command=self.start_analysis)
        self.run_button.grid(row=1, column=2, padx=(10, 0), pady=(5, 0))
        controls.columnconfigure(3, weight=1)
        ttk.Label(controls, text="Trading enabled: " + str(settings.trading_enabled), style="Muted.TLabel").grid(row=1, column=3, sticky="e")

        cards = ttk.Frame(self)
        cards.pack(fill="x", padx=24, pady=10)
        self.price_value = self._metric_card(cards, "PRICE")
        self.signal_value = self._metric_card(cards, "SIGNAL")
        self.confidence_value = self._metric_card(cards, "MODEL P(UP)")
        self.status_value = self._metric_card(cards, "STATUS")
        for i in range(4):
            cards.columnconfigure(i, weight=1)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=24, pady=(4, 24))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        chart_frame = ttk.Frame(body, style="Card.TFrame", padding=14)
        chart_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(chart_frame, text="Price history", font=("Segoe UI", 12, "bold"), background="#181d24", foreground="#ffffff").pack(anchor="w")
        self.chart = tk.Canvas(chart_frame, bg="#181d24", highlightthickness=0)
        self.chart.pack(fill="both", expand=True, pady=(10, 0))
        self.chart.bind("<Configure>", lambda _event: self._draw_chart())
        self._chart_data: list[float] = []

        info = ttk.Frame(body, style="Card.TFrame", padding=14)
        info.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(info, text="Quantitative reasoning", font=("Segoe UI", 12, "bold"), background="#181d24", foreground="#ffffff").pack(anchor="w")
        self.reasoning = tk.Text(info, height=7, wrap="word", bg="#181d24", fg="#d9e1e8", insertbackground="#ffffff", relief="flat", font=("Segoe UI", 10))
        self.reasoning.pack(fill="x", pady=(10, 18))
        self.reasoning.configure(state="disabled")

        ttk.Label(info, text="Run log", font=("Segoe UI", 11, "bold"), background="#181d24", foreground="#ffffff").pack(anchor="w")
        self.log = tk.Text(info, wrap="word", bg="#11161c", fg="#aeb9c5", insertbackground="#ffffff", relief="flat", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, pady=(8, 0))
        self.log.configure(state="disabled")
        self._set_status("Ready")

    def _metric_card(self, parent: ttk.Frame, title: str) -> ttk.Label:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=14)
        index = len(parent.winfo_children())
        frame.grid(row=0, column=index, sticky="nsew", padx=4)
        ttk.Label(frame, text=title, style="Metric.TLabel").pack(anchor="w")
        value = ttk.Label(frame, text="—", style="Value.TLabel")
        value.pack(anchor="w", pady=(5, 0))
        return value

    def _set_status(self, text: str) -> None:
        self.status_value.configure(text=text)

    def _write_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start_analysis(self) -> None:
        ticker = self.ticker_var.get().strip().upper()
        period = self.period_var.get().strip()
        if not ticker:
            messagebox.showwarning("BrokerIA", "Enter a ticker symbol.")
            return

        self.run_button.configure(state="disabled")
        self._set_status("Running…")
        self._append_log(f"Starting analysis: {ticker} / {period}")
        threading.Thread(target=self._analysis_worker, args=(ticker, period), daemon=True).start()

    def _analysis_worker(self, ticker: str, period: str) -> None:
        try:
            quote = get_stock_quote(ticker)
            data = add_indicators(get_historical_prices(ticker, period=period))
            signal = generate_signal(ticker, data, None, settings.model_threshold)
            prices = [float(v) for v in data["Close"].tail(180).dropna().tolist()]
            self.after(0, self._show_result, quote, signal, prices)
        except Exception as exc:
            self.after(0, self._show_error, str(exc))

    def _show_result(self, quote, signal, prices: list[float]) -> None:
        self.price_value.configure(text=f"${quote['price']:.2f}")
        self.signal_value.configure(text=signal.direction)
        self.confidence_value.configure(text=f"{signal.confidence:.1%}")
        self._set_status("Complete")
        self._write_text(self.reasoning, signal.reasoning)
        self._chart_data = prices
        self._draw_chart()
        self._append_log(f"Complete: {signal.direction} | P(up)={signal.confidence:.3f}")
        self.run_button.configure(state="normal")

    def _show_error(self, error: str) -> None:
        self._set_status("Error")
        self._append_log(f"ERROR: {error}")
        self.run_button.configure(state="normal")
        messagebox.showerror("Analysis failed", error)

    def _draw_chart(self) -> None:
        self.chart.delete("all")
        if len(self._chart_data) < 2:
            self.chart.create_text(20, 30, anchor="nw", text="Run an analysis to display the price chart.", fill="#7f8b98", font=("Segoe UI", 10))
            return
        width = max(self.chart.winfo_width(), 300)
        height = max(self.chart.winfo_height(), 250)
        pad = 35
        low, high = min(self._chart_data), max(self._chart_data)
        span = high - low or 1.0
        points = []
        for i, value in enumerate(self._chart_data):
            x = pad + i * (width - 2 * pad) / (len(self._chart_data) - 1)
            y = height - pad - (value - low) / span * (height - 2 * pad)
            points.extend((x, y))
        self.chart.create_line(*points, fill="#61dafb", width=2, smooth=False)
        self.chart.create_text(pad, height - 10, anchor="w", text=f"Low ${low:.2f}", fill="#7f8b98", font=("Segoe UI", 8))
        self.chart.create_text(width - pad, 10, anchor="e", text=f"High ${high:.2f}", fill="#7f8b98", font=("Segoe UI", 8))


def main() -> None:
    settings.validate_safety()
    app = BrokerIAApp()
    app.mainloop()


if __name__ == "__main__":
    main()

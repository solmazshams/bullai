        if iteration == 0:
            _, axes = plt.subplots(figsize=(8, 6), dpi = 300)
            cmap = plt.cm.get_cmap('tab20')
            num_colors = 20
            custom_colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
            axes.set_prop_cycle(cycler(color=custom_colors))

            for indicator in config["indicators"]:
                # scale = env.stocks[0].normalization_info[indicator][0]
                # bias = env.stocks[0].normalization_info[indicator][1]
                scale = 1
                bias = 0
                if indicator not in ["Close", "Open", "Low", "High", "ema_short", "ema_long", "bollinger_h", "bollinger_l"]:
                    print(f"{indicator:15s}",
                        f"\n\t max:  {scale * data[indicator].max() + bias:.2f}",
                        f"\n\t min:  {scale * data[indicator].min() + bias:.2f}",
                        f"\n\t mean: {scale * data[indicator].mean() + bias:.2f}"
                        )

                    axes.plot(scale * data[indicator] + bias, label = indicator, linewidth = 0.5)
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            axes.legend()
            plt.tight_layout()
            plt.savefig("plots/scaled_features.png", dpi = 300)
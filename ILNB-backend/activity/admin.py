from django.contrib import admin
from .models import TradeActivity


@admin.register(TradeActivity)
class TradeActivityAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'trade_type', 'asset_name', 'quantity', 'price',
        'total_amount', 'status', 'timestamp'
    )
    list_filter = ('trade_type', 'status', 'asset_type')
    search_fields = ('user__username', 'asset_name', 'trade_type')
    ordering = ('-timestamp',)
    readonly_fields = ('timestamp',)
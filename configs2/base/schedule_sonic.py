# optimizer
optimizer = dict(
    type='Adam',
    lr=5e-4,
)
auto_scale_lr = dict(enable=True, base_batch_size=16)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 18])

custom_hooks = [dict(type='SonicAfterRunHook')]

save_pipeline = [
    dict(
        type='SaveEachEpochModel',
        save_each_epoch=True,
        encrypt_each_epoch=False,
        save_latest=True,
        encrypt_latest=False),
    dict(type='SaveLatestModel', encrypt=False),
]

after_run_pipeline = [
    # dict(type='DeployModel'),
    dict(type='EncryptModel'),
    dict(type='SaveLog', create_briefing=True),
]

runner = dict(
    type='SonicEpochBasedRunner',
    save_pipeline=save_pipeline,
    after_run_pipeline=after_run_pipeline,
    max_epochs=12,
)

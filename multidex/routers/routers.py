def instrument_or_none(model):
    if hasattr(model, "instrument"):
        return getattr(model, "instrument")
    return None


class InstrumentRouter:
    """
    routes database calls from instrument models to instrument-specific
    databases. leaves other content (auth, etc.) untouched. pretty
    straightforward.
    """

    @staticmethod
    def db_for_read(model, **hints):
        """
        Attempts to read auth and contenttypes models go to auth_db.
        """
        return instrument_or_none(model)

    @staticmethod
    def db_for_write(model, **hints):
        return instrument_or_none(model)

    @staticmethod
    def allow_relation(model1, model2, **hints):
        if instrument_or_none(model1) == instrument_or_none(model2):
            return True
        return None

    #
    # def allow_migrate(self, db, app_label, model_name=None, **hints):
    #     if app_label in self.route_app_labels:
    #         return db == 'auth_db'
    #     return None
    #

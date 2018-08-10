from abc import *
from typing import *
import inspect
from functools import wraps
import numpy as np


###########################################################################
#                                                                         #
#             DO NOT FOCUS ON THIS META CLASS ON FIRST READ               #
#                                                                         #
###########################################################################


LIFECYCLE_BREAK_BEHAVIOR = 2  # 0 raise, 1 log-n-pass, 2 pass
CHECK_INTERFACE_BEHAVIOR = 0  # 0 method presence, 1 +property


class MetaAbstractActor(ABCMeta):
    """ The Metaclass of the Abstract Actor class. All weird stuffs might be contained here.

    We add a `__actorstate__` that contains an integer representing
    + 0: the agent is feshly created
    + 1: the agent is reset and can be used
    + 2: the agent is terminated and can't be used.

    Notes
    -----
    In particular, we add Automatic Wrapping approach from :
        http://www.voidspace.org.uk/python/articles/metaclasses.shtml#a-method-decorating-metaclass
    to check the state and transitions, along with a change in error message to improve readability.


    """

    def __new__(cls, classname, bases, classdict):
        """

        """

        # We wrap all methods
        wrapped_methods = ['__init__', 'reset', 'act', 'terminate']
        newclassdict = {}
        for name, attr in classdict.items():
            if inspect.isfunction(attr) and name in wrapped_methods:
                attr = MetaAbstractActor._wrapper(attr, classname)
            newclassdict[name] = attr

        # We add a state in dictionnary
        newclassdict['__actorstate__'] = 0

        # We perform static analysis of code.
        newclassdict['__actorast__'] = {}

        return ABCMeta.__new__(cls, classname, bases, newclassdict)

    @staticmethod
    def _wrapper(method, classname):
        """This decorator allows to wrap actor protocol methods with augmented checks and behviors.

        In particular the wrapped method does the following:
            + Check if method can be called and depending on selected behavior, pass, log or raise.
            + Put method in a try/except block, and modify the error message to be simpler to decypher.

        """

        # Authorized Calls and Transitions
        authorized_call_in_state = {0: ['__init__', 'reset'],
                                    1: ['reset', 'act', 'terminate'],
                                    2: []}
        state_after_call = {'__init__': 0,
                            'reset': 1,
                            'act': 1,
                            'terminate': 2}

        @wraps(method)  # Needed for doc to work.
        def wrapped(self, *args, **kwargs):
            """
            """

            # We check state
            assert self.__actorstate__ in [0, 1, 2], "Actor found in unknown state"
            if method.__name__ not in authorized_call_in_state[self.__actorstate__]:
                if LIFECYCLE_BREAK_BEHAVIOR == 0:
                    raise Exception("You broke the lifecycle of {} when calling {}. \
                                    Check the documentation.".format(classname, method.__name__))
                elif LIFECYCLE_BREAK_BEHAVIOR == 1:
                    print("You broke the lifecycle of {} when calling {}. \
                          Check the documentation.".format(classname, method.__name__))
                elif LIFECYCLE_BREAK_BEHAVIOR == 2:
                    pass

            # We execute method
            try:
                result = method(self, *args, **kwargs)
            except Exception as err:
                if len(err.args) == 0:
                    err.args = ('%s' % err.__class__.__name__,)
                message = '\nError Happened in {}.{} : '.format(classname, method.__name__)
                message += err.args[0]
                message = message.replace('\n', '\n\t')
                err.args = (message,) + err.args[1:]
                raise err

            # We change state
            self.__actorstate__ = state_after_call[method.__name__]

            return result

        return wrapped


class AbstractActor(metaclass=MetaAbstractActor):
    """ The Fundemental actor class, implementing our actor protocol.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Contains all the mechanics involved in the creation of the actor, and not its initialization.
        """

        pass

    @abstractmethod
    def reset(self, *args, **kwargs):
        """Contains all the necessary to (re-)initialize the agent to its initial state.
        """

        pass

    @abstractmethod
    def act(self, *args, **kwargs):
        """Contains the core of the actor function.
        """

        pass

    @abstractmethod
    def terminate(self):
        """Contains the necessary code to terminate the actor properly.
        """

        pass

    @classmethod
    @abstractmethod
    def test(cls):
        """Contains the necessary logic to test
        """

        pass


class IStaticEnvironment(metaclass=ABCMeta):
    """An Interface mixin representing a static environment.
    """

    @property
    @abstractmethod
    def observation(self) -> np.ndarray:
        pass


class IRewarding(metaclass=ABCMeta):
    """An Interface mixin representing a rewarding actor.
    """

    @property
    @abstractmethod
    def reward(self) -> float:
        pass


class IEpisodicEnvironment(metaclass=ABCMeta):
    """An Interface mixin representing an episodic environment.
    """

    @property
    @abstractmethod
    def observation_sequence(self) -> np.ndarray:
        pass


class IEpisodicRewarding(metaclass=ABCMeta):
    """An interface mixin representing an episodic rewarding environment.
    """

    @property
    @abstractmethod
    def reward_sequence(self) -> np.ndarray:
        pass


class IRenderer(metaclass=ABCMeta):
    """An Interface mixin representing a rendering actor.
    """

    @property
    @abstractmethod
    def rendering(self) -> np.ndarray:
        pass


class IController(metaclass=ABCMeta):
    """An interface mixin representing a controller.
    A controller is an actor returning an action sequence out of a (sparser) vector.
    """

    @property
    @abstractmethod
    def action_sequence(self) -> np.ndarray:
        pass


class ITrainable(metaclass=ABCMeta):
    """An interface mixin representing a trainable
    """

    @property
    @abstractmethod
    def prediction(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def performance(self):
        pass


class IParameterized(metaclass=ABCMeta):
    """An interface mixin representing a parameterized object
    """

    @property
    @abstractmethod
    def parameters(self) -> list:
        pass


class IDataset(ABC):
    """An interface mixin representing a dataset.
    """

    @property
    @abstractmethod
    def dataset(self):
        pass


class IRepresentation(metaclass=ABCMeta):
    """An Interface mixin representing a representation learning algorithm
    """

    @property
    @abstractmethod
    def representation(self) -> np.ndarray:
        pass


class IDistribution(metaclass=ABCMeta):
    """An interface mixin representing a distribution we sample from
    """

    @property
    @abstractmethod
    def sample(self) -> np.ndarray:
        pass


class IExplorer(metaclass=ABCMeta):
    """An interface mixin representing an exploration algorithm
    """

    @property
    @abstractmethod
    def environment(self) -> IStaticEnvironment:
        pass

    @property
    @abstractmethod
    def actions(self) -> List:
        pass

    @property
    @abstractmethod
    def outcomes(self) -> List:
        pass

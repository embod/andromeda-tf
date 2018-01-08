from uuid import UUID
from struct import pack_into, unpack_from
from websocket import create_connection


class Client:
    """

    """
    ADD_AGENT = bytes([0])
    REMOVE_AGENT = bytes([1])
    AGENT_ACTION = bytes([2])
    AGENT_STATE = bytes([3])

    ERROR = bytes([255])

    def __init__(self, apikey, state_callback):
        """

        :param apikey:
        :param state_callback:
        """
        self.websocket = create_connection("ws://dev.embod.ai:8080/v0/agent/control?apikey=%s" % apikey)

        self._state_callback = state_callback

        self.running = True

    def add_agent(self, agent_id):
        """

        :param agent_id:
        :return:
        """
        self.send_message(Client.ADD_AGENT, agent_id)
        pass

    def remove_agent(self, agent_id):
        """

        :param agent_id:
        :return:
        """
        self.send_message(Client.REMOVE_AGENT, agent_id)
        pass

    def send_agent_action(self, agent_id, action):
        """

        :return:
        """
        pass

    def run_loop(self):

        while self.running:
            data = self.websocket.recv()

            self.handle_message(data)


    def handle_message(self, data):
        """

        :return:
        """

        message_type, resource_id_bytes, message_size = unpack_from('c16si', data, 0)

        if message_size > 0:
            payload = unpack_from('%ds' % message_size, data, 21)



    def send_message(self, message_type, resource_id, data=None):
        """

        :param message_type:
        :param resource_id:
        :param data:
        :return:
        """
        payload_size = len(data)*4 if data is not None else 0
        buffer_size = 21+payload_size

        buffer = bytearray(buffer_size)

        pack_into('c', buffer, 0, message_type)
        pack_into('16s', buffer, 1, UUID(resource_id).bytes)

        if data is not None:
            pack_into('i', buffer, 17, payload_size)
            pack_into('f', buffer, 21, *data)
        else:
            pack_into('i', buffer, 17, 0)

        self.websocket.send_binary(buffer)